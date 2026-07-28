# -*- coding: utf-8 -*-
"""Tests for structured parallel progress and responsive GUI sizing."""

import io
from contextlib import redirect_stdout
from pathlib import Path
from queue import Queue
import sys
from threading import Lock
import unittest

from auto_reframe_core.gui import (
    calculate_responsive_window_geometry,
    format_video_event_log,
)
from auto_reframe_core.video_utils import (
    VideoProgressEvent,
    run_ffmpeg_with_progress,
    run_parallel,
)


class ResponsiveWindowGeometryTests(unittest.TestCase):
    def test_common_desktop_sizes_leave_room_for_system_chrome(self):
        self.assertEqual(
            calculate_responsive_window_geometry(1920, 1080),
            (1100, 900, 820, 620),
        )
        self.assertEqual(
            calculate_responsive_window_geometry(1366, 768),
            (1100, 648, 820, 620),
        )

    def test_small_display_never_requests_a_window_larger_than_the_screen(self):
        width, height, min_width, min_height = (
            calculate_responsive_window_geometry(800, 600)
        )

        self.assertEqual((width, height), (704, 520))
        self.assertLessEqual(min_width, width)
        self.assertLessEqual(min_height, height)


class VideoProgressEventTests(unittest.TestCase):
    def test_parallel_tasks_keep_lifecycle_events_associated_with_each_video(self):
        tasks = [
            (1, 2, Path("first.mp4")),
            (2, 2, Path("second.mp4")),
        ]
        events = []
        event_lock = Lock()

        def collect(event):
            with event_lock:
                events.append(event)

        result = run_parallel(
            tasks,
            lambda _task, _positions: True,
            workers=2,
            progress_callback=collect,
        )

        self.assertEqual(result, (2, []))
        for index, filename in ((1, "first.mp4"), (2, "second.mp4")):
            task_events = [event for event in events if event.index == index]
            self.assertEqual(task_events[0].kind, "queued")
            self.assertEqual(task_events[-1].kind, "completed")
            self.assertIn("started", [event.kind for event in task_events])
            self.assertTrue(all(event.filename == filename for event in task_events))

    def test_failed_task_has_a_terminal_failure_event(self):
        events = []

        result = run_parallel(
            [(1, 1, Path("broken.mp4"))],
            lambda _task, _positions: False,
            workers=1,
            progress_callback=events.append,
        )

        self.assertEqual(result, (0, ["broken.mp4"]))
        self.assertEqual(events[-1].kind, "failed")
        self.assertEqual(events[-1].message, "處理未完成")

    def test_ffmpeg_callback_replaces_repeated_text_progress(self):
        positions = Queue()
        positions.put(0)
        percentages = []
        captured = io.StringIO()
        command = [
            sys.executable,
            "-c",
            "print('frame=1 time=00:00:05.0')",
        ]

        with redirect_stdout(captured):
            returncode, _stderr = run_ffmpeg_with_progress(
                command,
                {"duration": 10.0},
                "structured-progress",
                positions,
                progress_callback=percentages.append,
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(percentages, [0, 50, 100])
        self.assertNotIn("[進度]", captured.getvalue())

    def test_gui_history_keeps_only_meaningful_lifecycle_events(self):
        progress = VideoProgressEvent(
            kind="progress",
            index=2,
            total=8,
            filename="demo.mp4",
            progress=65,
            phase="影片壓縮",
        )
        retry = VideoProgressEvent(
            kind="retry",
            index=2,
            total=8,
            filename="demo.mp4",
            progress=0,
            phase="影片壓縮",
            attempt=2,
            message="停用硬體解碼、保留硬體編碼",
        )

        self.assertEqual(format_video_event_log(progress), "")
        self.assertIn("[2/8] demo.mp4", format_video_event_log(retry))
        self.assertIn("第 2 次嘗試", format_video_event_log(retry))


if __name__ == "__main__":
    unittest.main()
