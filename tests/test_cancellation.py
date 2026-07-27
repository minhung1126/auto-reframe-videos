# -*- coding: utf-8 -*-

from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys
from threading import Event
import time
import unittest
from unittest.mock import patch

from auto_reframe_core.compress import VideoCompressor
from auto_reframe_core.reframe import VideoReframer
from auto_reframe_core.video_utils import (
    FFmpegCancellation,
    run_ffmpeg_with_progress,
    run_parallel,
)


class FFmpegCancellationTests(unittest.TestCase):
    def test_cancel_terminates_a_registered_child_process(self):
        cancellation = FFmpegCancellation()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            cancellation.register(process)
            cancellation.cancel(timeout=1.0)

            self.assertTrue(cancellation.cancelled)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_register_after_cancel_still_terminates_the_child(self):
        cancellation = FFmpegCancellation()
        cancellation.cancel()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            cancellation.register(process)

            self.assertIsNotNone(process.poll())
            self.assertEqual(cancellation.active_process_count, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_run_parallel_propagates_interrupt_and_cancels_running_workers(self):
        worker_started = Event()
        worker_observed_cancel = Event()

        def process_task(_task, position_queue):
            worker_started.set()
            if position_queue.cancellation.wait(timeout=2.0):
                worker_observed_cancel.set()
            return False

        def interrupt_after_worker_started(_futures):
            self.assertTrue(worker_started.wait(timeout=2.0))
            raise KeyboardInterrupt

        tasks = [(1, 1, Path("input.mp4"))]
        with patch(
            "auto_reframe_core.video_utils.as_completed",
            side_effect=interrupt_after_worker_started,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_parallel(tasks, process_task, workers=1)

        self.assertTrue(worker_observed_cancel.is_set())

    def test_cancelled_queued_video_tasks_do_not_start_ffprobe(self):
        position_queue = Queue()
        position_queue.cancellation = FFmpegCancellation()
        position_queue.cancellation.cancel()
        task = (1, 1, Path("input.mp4"))

        for processor_type, probe_target in (
            (VideoCompressor, "auto_reframe_core.compress.get_video_info"),
            (VideoReframer, "auto_reframe_core.reframe.get_video_info"),
        ):
            with self.subTest(processor=processor_type.__name__):
                processor = object.__new__(processor_type)
                with patch(probe_target) as get_video_info:
                    result = processor.process_single_video(task, position_queue)

                self.assertFalse(result)
                get_video_info.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_progress_runner_stops_an_active_ffmpeg_process(self):
        cancellation = FFmpegCancellation()
        position_queue = Queue()
        position_queue.cancellation = cancellation
        position_queue.put(0)
        command = [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x64:r=10:d=60",
            "-f",
            "null",
            "-",
        ]

        from concurrent.futures import ThreadPoolExecutor

        with patch("auto_reframe_core.video_utils._HAS_TQDM", False):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_ffmpeg_with_progress,
                    command,
                    {"duration": 60.0},
                    "cancel-test",
                    position_queue,
                )
                deadline = time.monotonic() + 3.0
                while (
                    cancellation.active_process_count == 0
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertEqual(cancellation.active_process_count, 1)

                cancellation.cancel(timeout=1.0)
                returncode, _stderr = future.result(timeout=3.0)

        self.assertNotEqual(returncode, 0)
        self.assertEqual(cancellation.active_process_count, 0)


if __name__ == "__main__":
    unittest.main()
