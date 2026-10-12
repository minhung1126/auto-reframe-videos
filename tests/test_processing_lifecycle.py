"""Output transactions, exception cleanup, and per-batch metadata contracts."""

from pathlib import Path
from queue import Queue
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.encoder_profiles import detect_target_encoders
from auto_reframe_core.output_plans import (
    build_compress_output_plan, build_reframe_output_plan,
    cleanup_temp_outputs, prepare_video_batch, promote_temp_outputs, validate_prepared_batch,
)
from auto_reframe_core.reframe import ReframeConfig, VideoReframer
from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
from auto_reframe_core.video_utils import FFmpegCancellation


INFO = {"width": 320, "height": 180, "fps": 30, "duration": 1, "has_audio": False}


class OutputPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.finals = [self.root / "first.mp4", self.root / "second.mp4"]
        self.tmps = [p.with_name(p.name + ".tmp") for p in self.finals]
        for index, (tmp, final) in enumerate(zip(self.tmps, self.finals)):
            tmp.write_bytes(f"new-{index}".encode())
            final.write_bytes(f"old-{index}".encode())

    def assert_originals_preserved(self):
        self.assertEqual([p.read_bytes() for p in self.finals], [b"old-0", b"old-1"])
        self.assertFalse(list(self.root.glob(".arv-promote-*")))

    def test_missing_or_empty_later_tmp_changes_no_final(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                if missing:
                    self.tmps[1].unlink()
                else:
                    self.tmps[1].write_bytes(b"")
                with self.assertRaisesRegex(ValueError, "暫存"):
                    promote_temp_outputs(self.tmps, self.finals)
                self.assert_originals_preserved()
                self.assertEqual(self.tmps[0].read_bytes(), b"new-0")

    def test_one_locked_tmp_does_not_prevent_cleanup_of_the_others(self):
        original_unlink = Path.unlink

        def locked_first(path, *args, **kwargs):
            if path == self.tmps[0]:
                raise PermissionError("locked tmp")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", locked_first):
            with self.assertRaises(PermissionError):
                cleanup_temp_outputs(self.tmps)
        self.assertTrue(self.tmps[0].exists())
        self.assertFalse(self.tmps[1].exists())

    def test_second_replace_failure_restores_the_entire_group(self):
        original_replace = Path.replace

        def fail_second(path, destination):
            if path == self.tmps[1]:
                raise OSError("replace failed")
            return original_replace(path, destination)

        with patch.object(Path, "replace", fail_second):
            with self.assertRaisesRegex(OSError, "replace failed"):
                promote_temp_outputs(self.tmps, self.finals)
        self.assert_originals_preserved()

    def test_first_replace_failure_does_not_delete_any_old_final(self):
        with patch.object(Path, "replace", side_effect=PermissionError("busy file")):
            with self.assertRaises(PermissionError):
                promote_temp_outputs(self.tmps, self.finals)
        self.assert_originals_preserved()

    def test_rollback_removes_outputs_that_did_not_exist_before(self):
        self.finals[0].unlink()
        original_replace = Path.replace

        def fail_second(path, destination):
            if path == self.tmps[1]:
                raise OSError("replace failed")
            return original_replace(path, destination)

        with patch.object(Path, "replace", fail_second):
            with self.assertRaises(OSError):
                promote_temp_outputs(self.tmps, self.finals)
        self.assertFalse(self.finals[0].exists())
        self.assertEqual(self.finals[1].read_bytes(), b"old-1")

    def test_cancel_between_replacements_rolls_back_the_first_output(self):
        cancellation = FFmpegCancellation()
        original_replace = Path.replace

        def cancel_after_first(path, destination):
            result = original_replace(path, destination)
            if path == self.tmps[0]:
                cancellation.cancel()
            return result

        with patch.object(Path, "replace", cancel_after_first):
            with self.assertRaises(InterruptedError):
                promote_temp_outputs(self.tmps, self.finals, cancellation=cancellation)
        self.assert_originals_preserved()

    def test_success_replaces_all_outputs_and_cleans_backups(self):
        with patch("auto_reframe_core.output_plans.os.link", side_effect=OSError("no links")):
            promote_temp_outputs(self.tmps, self.finals)
        self.assertEqual([p.read_bytes() for p in self.finals], [b"new-0", b"new-1"])
        self.assertTrue(all(not p.exists() for p in self.tmps))
        self.assertFalse(list(self.root.glob(".arv-promote-*")))

    def test_failed_rollback_preserves_recoverable_old_final(self):
        original_replace = Path.replace

        def fail_replace_and_rollback(path, destination):
            if path == self.tmps[1] or path.name.startswith(".arv-promote-"):
                raise OSError("locked")
            return original_replace(path, destination)

        with patch.object(Path, "replace", fail_replace_and_rollback):
            with self.assertRaisesRegex(RuntimeError, "備份保留"):
                promote_temp_outputs(self.tmps, self.finals)
        backups = list(self.root.glob(".arv-promote-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old-0")


class ProcessorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"original source")

    def processor(self, mode):
        config_cls, processor_cls = ((CompressConfig, VideoCompressor) if mode == "compress"
                                     else (ReframeConfig, VideoReframer))
        targets = [{"resolution": "source", "vcodec": "h264"}]
        if mode == "reframe":
            targets[0]["ratio"] = (4, 5)
        config = config_cls(video_files=(self.source,), output_dir=str(self.root / mode),
                            targets=targets, max_workers=1, skip_existing=False)
        processor = object.__new__(processor_cls)
        processor.config, processor.progress_callback = config, None
        processor._ffmpeg_attempts = Mock(return_value=[("test", ["fake-ffmpeg"])])
        output = Path(config.output_dir)
        if mode == "compress":
            plan = build_compress_output_plan(config, output, self.source, INFO)
        else:
            dims = calculate_reframe_dimensions(INFO["width"], INFO["height"], (4, 5), config.final_ratio)
            plan = build_reframe_output_plan(config, output, self.source, INFO, dims, 4, 5, targets)
        for final in plan.finals:
            final.write_bytes(b"existing final")
        return processor, plan

    def test_runner_and_progress_exceptions_clean_only_this_groups_tmp(self):
        for mode in ("compress", "reframe"):
            for callback_failure in (False, True):
                with self.subTest(mode=mode, callback_failure=callback_failure):
                    processor, plan = self.processor(mode)
                    unrelated = plan.tmps[0].parent / "unrelated.tmp"
                    unrelated.write_bytes(b"keep")
                    if callback_failure:
                        processor.progress_callback = Mock(side_effect=RuntimeError("callback failed"))

                    def fail_runner(*args, progress_callback=None, **kwargs):
                        plan.tmps[0].write_bytes(b"incomplete")
                        if progress_callback is not None:
                            progress_callback(25)
                        raise OSError("runner failed")

                    with patch(f"auto_reframe_core.{mode}.get_video_info", return_value=INFO), \
                         patch(f"auto_reframe_core.{mode}.run_ffmpeg_with_progress", side_effect=fail_runner):
                        with self.assertRaises((OSError, RuntimeError)):
                            processor.process_single_video((1, 1, self.source), Queue())
                    self.assertFalse(plan.tmps[0].exists())
                    self.assertEqual(plan.finals[0].read_bytes(), b"existing final")
                    self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_cancellation_after_success_prevents_promotion_and_cleans_tmp(self):
        for mode in ("compress", "reframe"):
            with self.subTest(mode=mode):
                processor, plan = self.processor(mode)
                positions = Queue()
                positions.cancellation = FFmpegCancellation()

                def finish_then_cancel(*args, **kwargs):
                    plan.tmps[0].write_bytes(b"completed")
                    positions.cancellation.cancel()
                    return 0, []

                with patch(f"auto_reframe_core.{mode}.get_video_info", return_value=INFO), \
                     patch(f"auto_reframe_core.{mode}.run_ffmpeg_with_progress", side_effect=finish_then_cancel):
                    self.assertFalse(processor.process_single_video((1, 1, self.source), positions))
                self.assertFalse(plan.tmps[0].exists())
                self.assertEqual(plan.finals[0].read_bytes(), b"existing final")

    def test_source_change_during_encoding_prevents_promotion(self):
        for mode in ("compress", "reframe"):
            with self.subTest(mode=mode):
                processor, plan = self.processor(mode)
                with patch("auto_reframe_core.video_utils.get_video_info", return_value=INFO):
                    processor.config.prepared_batch = prepare_video_batch(processor.config, mode)

                def finish_with_changed_source(*args, **kwargs):
                    plan.tmps[0].write_bytes(b"completed")
                    self.source.write_bytes(self.source.read_bytes() + b"changed")
                    return 0, []

                with patch(f"auto_reframe_core.{mode}.run_ffmpeg_with_progress", side_effect=finish_with_changed_source):
                    with self.assertRaisesRegex(ValueError, "來源影片.*變更"):
                        processor.process_single_video((1, 1, self.source), Queue())
                self.assertFalse(plan.tmps[0].exists())
                self.assertEqual(plan.finals[0].read_bytes(), b"existing final")

    def test_prepared_batch_and_workers_probe_each_source_only_once(self):
        for mode in ("compress", "reframe"):
            with self.subTest(mode=mode):
                processor, plan = self.processor(mode)

                def finish(*args, **kwargs):
                    plan.tmps[0].write_bytes(b"completed")
                    return 0, []

                with patch("auto_reframe_core.video_utils.get_video_info", return_value=INFO) as probe, \
                     patch(f"auto_reframe_core.{mode}.get_video_info", side_effect=AssertionError("reprobe")), \
                     patch(f"auto_reframe_core.{mode}.run_ffmpeg_with_progress", side_effect=finish):
                    processor.config.prepared_batch = prepare_video_batch(processor.config, mode)
                    validate_prepared_batch(processor.config, mode)
                    self.assertEqual(run_video_batch(processor.config, processor.process_single_video, "test"), (1, []))
                self.assertEqual(probe.call_count, 1)
                self.assertEqual(plan.finals[0].read_bytes(), b"completed")


class PreparedBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"original")
        self.config = CompressConfig(video_files=(self.source,), output_dir=str(self.root / "output"),
                                     targets=[{"resolution": "source", "vcodec": "h264"}])
        self.probe = patch("auto_reframe_core.video_utils.get_video_info", return_value=INFO)
        self.mock_probe = self.probe.start()
        self.addCleanup(self.probe.stop)
        self.config.prepared_batch = prepare_video_batch(self.config, "compress")

    def test_changed_sources_and_settings_are_refused_before_starting_workers(self):
        self.source.write_bytes(b"changed source")
        with patch("auto_reframe_core.batch_runner.run_parallel") as parallel:
            with self.assertRaisesRegex(ValueError, "來源影片.*變更"):
                run_video_batch(self.config, Mock(), "test")
            parallel.assert_not_called()
        self.config.targets[0]["vcodec"] = "h265"
        with self.assertRaisesRegex(ValueError, "設定.*變更"):
            validate_prepared_batch(self.config, "compress")
        self.assertEqual(self.mock_probe.call_count, 1)

    def test_skip_existing_can_change_without_invalidating_metadata(self):
        prepared = self.config.prepared_batch
        final = prepared.finals[0]
        final.write_bytes(b"existing")
        validate_prepared_batch(self.config, "compress")
        self.assertFalse(prepared.output_plan(self.source).has_work)
        self.config.skip_existing = False
        validate_prepared_batch(self.config, "compress")
        self.assertTrue(prepared.output_plan(self.source, skip_existing=False).has_work)
        self.assertEqual(self.mock_probe.call_count, 1)

    def test_later_directory_or_symlink_and_write_failures_are_revalidated(self):
        prepared = self.config.prepared_batch
        tmp = prepared.finals[0].with_name(prepared.finals[0].name + ".tmp")
        tmp.mkdir()
        with self.assertRaisesRegex(ValueError, "資料夾占用"):
            validate_prepared_batch(self.config, "compress")
        tmp.rmdir()
        original_is_symlink = Path.is_symlink

        def later_symlink(path):
            return path == tmp or original_is_symlink(path)

        with patch.object(Path, "is_symlink", later_symlink):
            with self.assertRaisesRegex(ValueError, "符號連結"):
                validate_prepared_batch(self.config, "compress")
        with patch("auto_reframe_core.output_plans.tempfile.NamedTemporaryFile", side_effect=PermissionError("read-only")):
            with self.assertRaises(PermissionError):
                validate_prepared_batch(self.config, "compress")
        self.assertEqual(self.mock_probe.call_count, 1)

    def test_source_change_during_probe_is_refused(self):
        def change_source(*args):
            self.source.write_bytes(b"modified while probing")
            return INFO

        self.mock_probe.side_effect = change_source
        with self.assertRaisesRegex(ValueError, "讀取資訊期間已變更"):
            prepare_video_batch(self.config, "compress")

    def test_changed_resource_selection_requires_new_preparation(self):
        self.config.watermark_width_ratio = 0.2
        with self.assertRaisesRegex(ValueError, "設定.*變更"):
            validate_prepared_batch(self.config, "compress")

    def test_unreadable_watermark_and_missing_reframe_font_are_rejected_before_probe(self):
        self.config.watermark_enabled = True
        self.config.watermark_file = str(self.root / "missing.png")
        with self.assertRaisesRegex(ValueError, "找不到浮水印"):
            prepare_video_batch(self.config, "compress")
        config = ReframeConfig(video_files=(self.source,), font_path=str(self.root / "missing.ttf"))
        with self.assertRaisesRegex(ValueError, "找不到字型"):
            prepare_video_batch(config, "reframe")
        self.assertEqual(self.mock_probe.call_count, 1)


class SelectedEncoderTests(unittest.TestCase):
    def test_single_codec_and_multi_codec_share_one_encoder_listing(self):
        for codecs in (("h264",), ("h264", "h265")):
            with self.subTest(codecs=codecs):
                listing = subprocess.CompletedProcess([], 0, stdout="h264_nvenc\nhevc_nvenc", stderr="")
                success = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                with patch("auto_reframe_core.encoder_profiles.subprocess.run", side_effect=[listing] + [success] * len(codecs)) as run:
                    profiles = detect_target_encoders("ffmpeg", codecs)
                self.assertEqual(set(profiles), set(codecs))
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(sum("-encoders" in command for command in commands), 1)
                if codecs == ("h264",):
                    self.assertFalse(any("hevc_nvenc" in command for command in commands))

    def test_both_processors_probe_only_the_selected_codec(self):
        for mode in ("compress", "reframe"):
            with self.subTest(mode=mode):
                config_cls, processor_cls = ((CompressConfig, VideoCompressor) if mode == "compress"
                                             else (ReframeConfig, VideoReframer))
                target = {"resolution": "source", "vcodec": "h264"}
                kwargs = {}
                if mode == "reframe":
                    target["ratio"] = (4, 5)
                    kwargs = {"top_text_override": "", "bottom_text_override": ""}
                with patch(f"auto_reframe_core.{mode}.detect_target_encoders", return_value={"h264": ("libx264", None)}) as detect:
                    processor_cls(config_cls(targets=[target], **kwargs))
                self.assertEqual(detect.call_args.args[1], {"h264"})


if __name__ == "__main__":
    unittest.main()
