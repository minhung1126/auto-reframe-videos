"""Synthetic media verifies preview geometry, final results and GUI actions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.preview import render_preview, PreviewService
from auto_reframe_core.process_runner import TaskCancelled
from auto_reframe_core.job_models import (
    TaskContext,
    ResultStatus,
    BatchResult,
    VideoResult,
    JobRequest,
)
from auto_reframe_core.gui_results_view import ResultsView
from auto_reframe_core.video_utils import get_video_info
from auto_reframe_core.output_plans import promote_temp_outputs

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg media integration"
)
class SyntheticMediaWorkflows(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="arv-test-中文 空白-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "來源 '中文.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=10:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.source),
            ],
            check=True,
            timeout=15,
        )

    def test_preview_uses_formal_dimensions_and_does_not_create_output_folders(self):
        watermark = self.root / "preview-watermark.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=16x8:duration=1,format=rgba",
                "-frames:v",
                "1",
                "-threads",
                "1",
                str(watermark),
            ],
            check=True,
            timeout=15,
        )
        for mode, cls in (("reframe", ReframeConfig), ("compress", CompressConfig)):
            kwargs = (
                {
                    "font_path": str(ROOT / "fonts/NotoSerifTC.ttf"),
                    "top_text_override": "繁體中文 100%\n上方",
                    "bottom_text_override": "下方",
                }
                if mode == "reframe"
                else {}
            )
            config = cls(
                video_files=(self.source,),
                output_dir=str(self.root / "never-created"),
                watermark_enabled=True,
                watermark_file=str(watermark),
                **kwargs,
            )
            data, size = render_preview(mode, config, self.source, 0.2)
            image = self.root / f"{mode}-preview.png"
            image.write_bytes(data)
            info = get_video_info("ffprobe", image)
            self.assertEqual((info["width"], info["height"]), size)
            self.assertFalse(Path(config.output_dir).exists())
            self.assertGreater(len(data), 100)

    def test_success_then_skip_and_required_player_container_flags(self):
        audio_source = self.root / "audio-source.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(self.source),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(audio_source),
            ],
            check=True,
            timeout=15,
        )
        config = CompressConfig(
            video_files=(audio_source,),
            output_dir=str(self.root / "output"),
            targets=[
                {"resolution": "source", "vcodec": "h264"},
                {"resolution": "source", "vcodec": "h265"},
            ],
            max_workers=1,
        )
        with patch(
            "auto_reframe_core.compress.detect_target_encoders",
            return_value={"h264": ("libx264", None), "h265": ("libx265", None)},
        ):
            processor = VideoCompressor(config)
        first = processor.run_result()
        second = processor.run_result()
        self.assertEqual(first.count(ResultStatus.SUCCESS), 1)
        self.assertEqual(second.count(ResultStatus.SKIPPED), 1)
        self.assertEqual(first.videos[0].targets[0].status, ResultStatus.SUCCESS)
        final = first.videos[0].targets[0].path
        media = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(final),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        info = json.loads(media.stdout)
        video = next(
            stream for stream in info["streams"] if stream["codec_type"] == "video"
        )
        self.assertEqual(video["codec_name"], "h264")
        self.assertEqual(video["pix_fmt"], "yuv420p")
        payload = final.read_bytes()
        self.assertLess(payload.index(b"moov"), payload.index(b"mdat"))
        self.assertFalse(list(final.parent.glob("*.tmp")))
        for target in first.videos[0].targets:
            container = json.loads(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-show_streams",
                        "-of",
                        "json",
                        str(target.path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                ).stdout
            )
            audio = next(
                stream
                for stream in container["streams"]
                if stream["codec_type"] == "audio"
            )
            self.assertEqual(audio["codec_name"], "aac")
            video = next(
                stream
                for stream in container["streams"]
                if stream["codec_type"] == "video"
            )
            if video["codec_name"] == "hevc":
                self.assertEqual(video["codec_tag_string"], "hvc1")
            payload = target.path.read_bytes()
            self.assertLess(payload.index(b"moov"), payload.index(b"mdat"))

    def test_drag_drop_failure_keeps_one_usable_gui_interpreter(self):
        import tkinter

        try:
            import tkinterdnd2

            check = tkinter.Tk()
            check.destroy()
        except (ImportError, tkinter.TclError):
            self.skipTest("Tk display is required")
        from auto_reframe_core.gui_style import create_gui_root
        from auto_reframe_core.gui import AutoReframeGUI

        with patch(
            "tkinterdnd2.TkinterDnD._require",
            side_effect=RuntimeError("test load failure"),
        ):
            root = create_gui_root()
        try:
            root.withdraw()
            app = AutoReframeGUI(root, data_root=self.root / "fallback-data")
            app.output_subfolder_var.set("manual input works")
            self.assertEqual(app.output_subfolder_var.get(), "manual input works")
            self.assertEqual(str(app.output_subfolder_var._tk), str(root.tk))
            labels = []

            def collect(widget):
                try:
                    labels.append(str(widget.cget("text")))
                except tkinter.TclError:
                    pass
                for child in widget.winfo_children():
                    collect(child)

            collect(app.files_tab)
            self.assertTrue(any("拖曳無法使用" in text for text in labels))
        finally:
            root.destroy()

    def test_real_gui_stop_button_keeps_ui_responsive_during_preflight(self):
        import tkinter
        from threading import Event

        try:
            from tkinterdnd2 import TkinterDnD

            root = TkinterDnD.Tk()
        except (ImportError, RuntimeError, tkinter.TclError):
            self.skipTest("Qualified GUI runtime is required by the CI gate")
        from auto_reframe_core.gui import AutoReframeGUI

        entered = Event()
        try:
            root.withdraw()
            app = AutoReframeGUI(root, data_root=self.root / "gui-stop-data")
            app.video_files = (self.source,)

            def slow_probe(request, context):
                entered.set()
                context.cancellation.wait(5)
                context.check()

            with patch.object(app.batch_service, "prepare", side_effect=slow_probe):
                app.start_button.invoke()
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                app.stop_button.invoke()
                root.update()
                self.assertLess(time.monotonic() - started, 0.2)
                deadline = time.monotonic() + 3
                while app.running and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertFalse(app.running)
                self.assertIn("已取消", app.status_var.get())
                self.assertEqual(
                    app.job_controller.context.cancellation.active_process_count, 0
                )
        finally:
            root.destroy()

    def test_actual_complete_gui_under_tk_and_drag_drop(self):
        import tkinter

        try:
            from tkinterdnd2 import TkinterDnD

            root = TkinterDnD.Tk()
            root.destroy()
        except (ImportError, RuntimeError, tkinter.TclError):
            self.skipTest(
                "Interactive GUI runtime qualification is a separate mandatory CI gate"
            )
        from auto_reframe_core.gui_smoke import exercise_gui

        report = exercise_gui(self.source, self.root)
        self.assertIn("reframe", report["gui_operations"])
        self.assertIn("compress", report["gui_operations"])
        self.assertTrue(report["tk"].startswith("8."))


class LatestPreviewTests(unittest.TestCase):
    def test_continuous_changes_cancel_active_request_and_only_deliver_latest(self):
        from threading import Event

        entered = Event()
        messages = []

        def render(*args, task_context):
            if args[0] == "first":
                entered.set()
                self.assertTrue(task_context.cancellation.wait(2))
                raise TaskCancelled()
            return b"png", (10, 20)

        with patch("auto_reframe_core.preview.render_preview", side_effect=render):
            service = PreviewService(messages.append)
            try:
                service.request("first")
                self.assertTrue(entered.wait(2))
                for index in range(20):
                    last = service.request(f"change-{index}")
                deadline = time.monotonic() + 3
                while service.busy and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(service.busy)
                self.assertTrue(messages)
                self.assertEqual(messages[-1].job_id, last)
                self.assertTrue(
                    all(message.kind == "preview_done" for message in messages)
                )
            finally:
                service.cancel(close=True)
                service.worker.join(2)

    def test_retry_only_selected_failed_sources_and_never_cancelled_sources(self):
        view = ResultsView()
        view.running = view.scan_busy = view.update_busy = False
        sources = tuple(Path(f"{index}.mp4") for index in range(4))
        view.last_result = BatchResult(
            "old",
            tuple(
                VideoResult(source, status)
                for source, status in zip(
                    sources,
                    (
                        ResultStatus.SUCCESS,
                        ResultStatus.FAILED,
                        ResultStatus.FAILED,
                        ResultStatus.CANCELLED,
                    ),
                )
            ),
        )
        view.last_request = JobRequest(
            "compress", CompressConfig(video_files=sources, prepared_batch=object())
        )
        view.progress_tree = Mock()
        view.progress_tree.selection.return_value = ("video-3", "video-4")
        view.start_job = Mock()
        view.retry_failed()
        mode, config = view.start_job.call_args.kwargs["request_override"]
        self.assertEqual(mode, "compress")
        self.assertEqual(config.video_files, (sources[2],))
        self.assertIsNone(config.prepared_batch)
        self.assertEqual(view.last_request.config.video_files, sources)


if __name__ == "__main__":
    unittest.main()
