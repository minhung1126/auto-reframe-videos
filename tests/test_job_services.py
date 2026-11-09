"""Real child processes and filesystem boundaries for the refactored services."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import errno
import json
from pathlib import Path
from queue import Empty
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch

from auto_reframe_core.batch_runner import run_video_batch_result
from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.config_migration import migrate_config, MIGRATION_RELEASES
from auto_reframe_core.config_store import (
    ConfigSession,
    ConfigConflictError,
    ConfigStoreError,
    load_config,
    save_config,
)
from auto_reframe_core.diagnostics import diagnostic_report
from auto_reframe_core.gui_settings import normalize_gui_settings
from auto_reframe_core.job_controller import EventBuffer, JobController, JobState
from auto_reframe_core.job_models import (
    JobEvent,
    TaskContext,
    ResultStatus,
    BatchResult,
    VideoResult,
)
from auto_reframe_core.output_locks import OutputLocks, OutputBusyError
from auto_reframe_core.output_plans import (
    normalize_output_subfolder,
    prepare_video_batch,
    validate_prepared_batch,
    check_batch_space,
    promote_temp_outputs,
)
from auto_reframe_core.process_runner import ProcessRunner, TaskCancelled
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.video_list import collect_videos
from auto_reframe_core.video_utils import (
    VideoProgressEvent,
    run_parallel,
    get_video_info,
)

ROOT = Path(__file__).resolve().parents[1]
INFO = {"width": 320, "height": 180, "fps": 30, "duration": 1, "has_audio": False}


class OwnedProcessesTests(unittest.TestCase):
    def test_probe_cancellation_reaps_process_within_five_seconds(self):
        context = TaskContext()
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(
                ProcessRunner(context).capture,
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )
            deadline = time.monotonic() + 3
            while (
                not context.cancellation.active_process_count
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(context.cancellation.active_process_count, 1)
            started = time.monotonic()
            context.cancellation.cancel(timeout=0.5)
            with self.assertRaises(TaskCancelled):
                future.result(timeout=4)
            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(context.cancellation.active_process_count, 0)

    def test_probe_timeout_also_reaps_process(self):
        context = TaskContext()
        with self.assertRaises(subprocess.TimeoutExpired):
            ProcessRunner(context).capture(
                [sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.1
            )
        self.assertEqual(context.cancellation.active_process_count, 0)

    def test_stall_is_visible_and_is_not_a_hard_timeout(self):
        status, percentages = [], []
        command = [
            sys.executable,
            "-c",
            "import time; print('out_time_us=500000', flush=True); time.sleep(.25); print('progress=end', flush=True)",
        ]
        code, stderr = ProcessRunner(TaskContext()).ffmpeg(
            command,
            {"duration": 1},
            percentages.append,
            status=status.append,
            stall_after=0.05,
        )
        self.assertEqual(code, 0)
        self.assertIn(50, percentages)
        self.assertTrue(any("暫未收到進度" in item for item in status))
        self.assertIn("已恢復處理進度", status)

    def test_concurrent_progress_and_large_stderr_cannot_deadlock(self):
        command = [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('diagnostic\\n' * 20000); print('out_time_us=1000000', flush=True)",
        ]
        code, tail = ProcessRunner(TaskContext()).ffmpeg(command, {"duration": 1})
        self.assertEqual(code, 0)
        self.assertLessEqual(len(tail), 30)

    def test_cancelled_scan_and_preflight_do_not_probe(self):
        context = TaskContext()
        context.cancellation.request()
        with self.assertRaises(TaskCancelled):
            collect_videos([ROOT], recursive=True, task_context=context)
        with patch("auto_reframe_core.video_utils.get_video_info") as probe:
            config = CompressConfig(video_files=(Path("not-read.mp4"),))
            with self.assertRaises(TaskCancelled):
                prepare_video_batch(config, "compress", task_context=context)
            probe.assert_not_called()


class CrossProcessOwnershipTests(unittest.TestCase):
    def test_two_independent_processes_cannot_silently_overwrite_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_config(path, {"workers": 1})
            script = """from auto_reframe_core.config_store import ConfigSession, ConfigConflictError
import sys
session=ConfigSession(sys.argv[1]); session.load(); print('ready', flush=True); sys.stdin.readline()
try:
    session.save({'workers':int(sys.argv[2])}); print('saved', flush=True)
except ConfigConflictError:
    print('conflict', flush=True)
"""
            children = [
                subprocess.Popen(
                    [sys.executable, "-u", "-c", script, str(path), str(value)],
                    cwd=ROOT,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for value in (2, 3)
            ]
            try:
                for child in children:
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                for child in children:
                    child.stdin.write("go\n")
                    child.stdin.flush()
                outcomes = [child.stdout.readline().strip() for child in children]
                self.assertCountEqual(outcomes, ["saved", "conflict"])
                self.assertIn(load_config(path)["workers"], (2, 3))
            finally:
                for child in children:
                    child.communicate(timeout=5)

    def test_second_process_cannot_acquire_and_crash_releases_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "輸出 空白.mp4"
            script = "from auto_reframe_core.output_locks import OutputLocks; import sys; lock=OutputLocks([sys.argv[1]]).acquire(); print('locked', flush=True); sys.stdin.readline()"
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", script, str(output)],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(OutputBusyError, "其他工作"):
                    OutputLocks([output]).acquire()
                # Unrelated outputs are still independently usable.
                with OutputLocks([output.with_name("other.mp4")]):
                    pass
                process.kill()
                process.wait(timeout=3)
                with OutputLocks([output]):
                    pass
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=3)

    def test_conflicting_job_never_cleans_another_jobs_tmp(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            source.write_bytes(b"source")
            config = CompressConfig(
                video_files=(source,), output_dir=str(Path(temporary) / "output")
            )
            with patch(
                "auto_reframe_core.video_utils.get_video_info", return_value=INFO
            ):
                config.prepared_batch = prepare_video_batch(config, "compress")
            final = config.prepared_batch.finals[0]
            tmp = final.with_name(final.name + ".tmp")
            tmp.write_bytes(b"other job owns this")
            with OutputLocks(config.prepared_batch.finals):
                with self.assertRaises(OutputBusyError):
                    run_video_batch_result(config, lambda *_: True, "test")
            self.assertEqual(tmp.read_bytes(), b"other job owns this")

    def test_config_sessions_and_independent_process_detect_stale_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_config(path, {"workers": 1})
            session = ConfigSession(path)
            session.load()
            script = "from auto_reframe_core.config_store import ConfigSession; import sys; s=ConfigSession(sys.argv[1]); s.load(); s.save({'workers':2})"
            subprocess.run(
                [sys.executable, "-c", script, str(path)],
                cwd=ROOT,
                check=True,
                timeout=5,
            )
            with self.assertRaises(ConfigConflictError):
                session.save({"workers": 3})
            with self.assertRaises(ConfigConflictError):
                session.clear()
            self.assertEqual(load_config(path)["workers"], 2)
            self.assertFalse(list(path.parent.glob(".config-*.tmp")))

    def test_disk_full_during_save_keeps_original_and_cleans_unique_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_config(path, {"workers": 1})
            with patch(
                "auto_reframe_core.config_store.os.fsync",
                side_effect=OSError(errno.ENOSPC, "full"),
            ):
                with self.assertRaises(ConfigStoreError):
                    save_config(path, {"workers": 2})
            self.assertEqual(load_config(path), {"workers": 1})
            self.assertFalse(list(path.parent.glob(".config-*.tmp")))

    def test_recover_preserves_corrupt_bytes_and_unknown_version_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            original = b"broken utf8 \xff"
            path.write_bytes(original)
            session = ConfigSession(path)
            with self.assertRaises(ConfigStoreError):
                session.load()
            backup = session.recover()
            self.assertEqual(backup.read_bytes(), original)
            session.save({"workers": 0})
            path.write_text('{"version":99,"settings":{}}')
            with self.assertRaisesRegex(ConfigStoreError, "不支援的設定檔版本"):
                session.load()


class ConfigMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "legacy/config.json"
        self.new = self.root / "Application Support/config.json"
        self.defaults = load_config(ROOT / "config.json.example")
        save_config(self.old, {"top_text": "繁體中文\n\n文字", "workers": 2})

    def migrate(self, **kwargs):
        return migrate_config(
            self.new, [self.old], self.defaults, normalize_gui_settings, **kwargs
        )

    def test_migration_is_a_verified_move_with_default_overlay(self):
        self.assertEqual(self.migrate(), self.old)
        self.assertFalse(self.old.exists())
        settings = load_config(self.new)
        self.assertEqual(settings["top_text"], "繁體中文\n\n文字")
        self.assertEqual(settings["workers"], 2)
        self.assertEqual(set(settings), set(self.defaults))
        self.assertIsNone(self.migrate())

    def test_exactly_five_versions_then_old_location_is_ignored(self):
        self.assertEqual(len(MIGRATION_RELEASES), 5)
        for version in [(3, 0, 13), (3, 0, 19), (3, 1, 0)]:
            self.assertIsNone(self.migrate(version=version))
            self.assertTrue(self.old.exists())
            self.assertFalse(self.new.exists())
        self.assertEqual(self.migrate(version=(3, 0, 18)), self.old)

    def test_existing_new_config_wins_and_both_originals_are_preserved(self):
        save_config(self.new, {"workers": 7})
        self.assertIsNone(self.migrate())
        self.assertEqual(load_config(self.new), {"workers": 7})
        self.assertTrue(self.old.exists())

    def test_invalid_old_document_never_moves_or_creates_new_file(self):
        save_config(self.old, {"workers": -1})
        with self.assertRaises(ConfigStoreError):
            self.migrate()
        self.assertTrue(self.old.exists())
        self.assertFalse(self.new.exists())

    def test_failed_deletion_rolls_back_new_copy_and_can_retry(self):
        original = Path.unlink

        def fail_source(path, *args, **kwargs):
            if path == self.old:
                raise PermissionError("read-only old directory")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_source):
            with self.assertRaises(PermissionError):
                self.migrate()
        self.assertTrue(self.old.exists())
        self.assertFalse(self.new.exists())
        self.migrate()
        self.assertFalse(self.old.exists())

    def test_legacy_text_and_custom_font_paths_survive_move(self):
        save_config(self.old, {"font_path": "fonts/custom.ttf"})
        (self.old.parent / "top_text.txt").write_text(
            "\ufeff標題\r\n\r\n第二段\n", encoding="utf-8"
        )
        self.migrate()
        settings = load_config(self.new)
        self.assertEqual(settings["top_text"], "標題\n\n第二段")
        self.assertEqual(
            settings["font_path"], str((self.old.parent / "fonts/custom.ttf").resolve())
        )
        self.assertTrue((self.old.parent / "top_text.txt").exists())

    def test_selected_legacy_watermark_is_copied_and_original_is_retained(self):
        artwork = self.old.parent / "watermark" / "test.png"
        artwork.parent.mkdir()
        artwork.write_bytes(b"local PNG fixture")
        save_config(
            self.old, {"watermarks": {"reframe": {"enabled": True, "file": "test.png"}}}
        )
        self.migrate()
        self.assertEqual(
            (self.new.parent / "watermark/test.png").read_bytes(), artwork.read_bytes()
        )
        self.assertTrue(artwork.exists())
        self.assertEqual(
            load_config(self.new)["watermarks"]["reframe"]["file"], "test.png"
        )


class ResultsAndControllerTests(unittest.TestCase):
    def test_cancel_before_encoder_initialization_releases_reserved_outputs(self):
        from auto_reframe_core.batch_runner import BatchService
        from auto_reframe_core.job_models import JobRequest
        from unittest.mock import Mock

        context = TaskContext()
        lease = context.output_lease = Mock()
        context.cancellation.request()
        with self.assertRaises(TaskCancelled):
            BatchService().execute(JobRequest("compress", CompressConfig()), context)
        lease.release.assert_called_once()

    def test_cancelled_queue_keeps_prior_success_and_does_not_start_later_tasks(self):
        context = TaskContext()
        calls = []

        def process(task, _positions):
            calls.append(task[2])
            if task[0] == 2:
                context.cancellation.request()
                return False
            return True

        tasks = [(index, 3, Path(f"{index}.mp4")) for index in (1, 2, 3)]
        result = run_parallel(tasks, process, 1, task_context=context, detailed=True)
        self.assertEqual(
            [video.status for video in result.videos],
            [ResultStatus.SUCCESS, ResultStatus.CANCELLED, ResultStatus.CANCELLED],
        )
        self.assertEqual(calls, [Path("1.mp4"), Path("2.mp4")])
        self.assertEqual(result.status, ResultStatus.CANCELLED)

    def test_controller_rejects_invalid_operations_and_stale_or_duplicate_completion(
        self,
    ):
        controller = JobController()
        context = controller.begin(JobState.PREPARING)
        with self.assertRaises(RuntimeError):
            controller.begin(JobState.SCANNING)
        controller.transition(JobState.CONFIRMING)
        controller.transition(JobState.VALIDATING)
        controller.transition(JobState.RUNNING)
        self.assertTrue(controller.accepts(JobEvent(context.job_id, "done")))
        self.assertTrue(controller.finish())
        self.assertFalse(controller.finish())
        self.assertFalse(controller.accepts(JobEvent(context.job_id, "done")))
        controller.begin(JobState.SCANNING)
        self.assertFalse(controller.accepts(JobEvent(context.job_id, "scan_done")))

    def test_cancel_button_state_is_immediate_and_never_waits_on_reaping(self):
        controller = JobController()
        controller.begin(JobState.PREPARING)
        with patch.object(
            controller.context.cancellation,
            "cancel",
            side_effect=lambda: time.sleep(0.3),
        ):
            started = time.monotonic()
            controller.cancel(close=True)
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertEqual(controller.state, JobState.CANCELLING)
            self.assertTrue(controller.close_when_finished)

    def test_thousand_jobs_have_bounded_progress_and_lossless_terminal_delivery(self):
        buffer = EventBuffer(maxsize=128, progress_limit=32)
        completed = []

        def produce():
            for index in range(1000):
                for percent in range(20):
                    event = VideoProgressEvent(
                        "progress", index, 1000, "video.mp4", percent
                    )
                    buffer.put(JobEvent("job", "video_progress", event))
                buffer.put(JobEvent("job", "terminal", index))

        producer = Thread(target=produce)
        producer.start()
        deadline = time.monotonic() + 5
        while producer.is_alive() or buffer.qsize():
            self.assertLessEqual(buffer.qsize(), 160)
            self.assertLess(time.monotonic(), deadline)
            try:
                item = buffer.get_nowait()
                if item.kind == "terminal":
                    completed.append(item.payload)
            except Empty:
                time.sleep(0.001)
        producer.join()
        self.assertEqual(completed, list(range(1000)))

    def test_diagnostic_export_contains_no_personal_source_or_text(self):
        result = BatchResult(
            "job",
            (
                VideoResult(
                    Path("/private/person/secret-title.mp4"),
                    ResultStatus.FAILED,
                    error_kind="disk_full",
                    message="private overlay text",
                ),
            ),
        )
        payload = json.dumps(diagnostic_report(result))
        for value in ("/private", "secret-title", "private overlay"):
            self.assertNotIn(value, payload)
        self.assertIn("disk_full", payload)


class OutputSubfolderTests(unittest.TestCase):
    def test_rejects_paths_reserved_names_and_control_characters(self):
        for value in (
            "../escape",
            "/tmp/escape",
            "..",
            ".",
            "a/b",
            "a\\b",
            "C:\\out",
            "CON",
            "LPT1.txt",
            "nul.png",
            "bad\nname",
            "trailing.",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_output_subfolder(value)
        self.assertEqual(normalize_output_subfolder("  中文 活動's  "), "中文 活動's")
        self.assertEqual(normalize_output_subfolder(""), "")

    def test_both_modes_and_destinations_include_name_and_snapshot_rejects_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            source.write_bytes(b"source")
            for cls, mode in ((CompressConfig, "compress"), (ReframeConfig, "reframe")):
                for output_mode in ("source", "specified"):
                    config = cls(
                        video_files=(source,),
                        output_dir=str(source.parent / "exports"),
                        output_mode=output_mode,
                        output_subfolder="中文 場次",
                    )
                    with patch(
                        "auto_reframe_core.video_utils.get_video_info",
                        return_value=INFO,
                    ):
                        config.prepared_batch = prepare_video_batch(config, mode)
                    self.assertTrue(
                        all(
                            "中文 場次" in path.parts
                            for path in config.prepared_batch.finals
                        )
                    )
                    config.output_subfolder = "other"
                    with self.assertRaisesRegex(ValueError, "設定.*變更"):
                        validate_prepared_batch(config, mode)


class TransactionRecoveryTests(unittest.TestCase):
    def test_disk_full_before_backup_copy_keeps_old_final(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "final.mp4"
            tmp = Path(temporary) / "final.mp4.tmp"
            final.write_bytes(b"original")
            tmp.write_bytes(b"new")
            with (
                patch(
                    "auto_reframe_core.output_plans.os.link",
                    side_effect=OSError("no links"),
                ),
                patch(
                    "auto_reframe_core.output_plans.shutil.disk_usage",
                    return_value=SimpleNamespace(free=0),
                ),
            ):
                with self.assertRaises(OSError) as error:
                    promote_temp_outputs([tmp], [final])
            self.assertEqual(error.exception.errno, errno.ENOSPC)
            self.assertEqual(final.read_bytes(), b"original")
            self.assertEqual(tmp.read_bytes(), b"new")
            self.assertFalse(list(final.parent.glob(".arv-promote-*")))

    def test_crash_journal_blocks_only_affected_outputs_and_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            source.write_bytes(b"source")
            config = CompressConfig(
                video_files=(source,), output_dir=str(source.parent / "out")
            )
            with patch(
                "auto_reframe_core.video_utils.get_video_info", return_value=INFO
            ):
                prepared = prepare_video_batch(config, "compress")
                final = prepared.finals[0]
                journal = final.parent / ".arv-transaction-crashed.json"
                journal.write_text(json.dumps({"outputs": [str(final)]}))
                with self.assertRaisesRegex(ValueError, "未完成的輸出交易"):
                    prepare_video_batch(config, "compress")
                self.assertTrue(journal.exists())
                journal.write_text(
                    json.dumps({"outputs": [str(final.with_name("unrelated.mp4"))]})
                )
                prepare_video_batch(config, "compress")
                self.assertTrue(journal.exists())

    def test_only_transient_windows_sharing_errors_receive_finite_retries(self):
        from unittest.mock import Mock
        from auto_reframe_core.job_errors import replace_with_retry

        sharing = PermissionError("temporarily busy")
        sharing.winerror = 32
        source = Mock()
        source.replace.side_effect = [sharing, None]
        with patch("auto_reframe_core.job_errors.time.sleep"):
            replace_with_retry(source, Path("final"))
        self.assertEqual(source.replace.call_count, 2)
        source.replace.reset_mock()
        source.replace.side_effect = OSError(errno.ENOSPC, "full")
        with self.assertRaises(OSError):
            replace_with_retry(source, Path("final"))
        self.assertEqual(source.replace.call_count, 1)


if __name__ == "__main__":
    unittest.main()
