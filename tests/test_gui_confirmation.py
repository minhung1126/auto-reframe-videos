# -*- coding: utf-8 -*-
"""Tests for the pre-processing confirmation summary."""

import unittest
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_reframe_core.compress import CompressConfig
from auto_reframe_core.gui_dialogs import (
    JobConfirmationDialog,
    OutputConflictDialog,
    OUTPUT_CONFLICT_CANCEL,
    OUTPUT_CONFLICT_OVERWRITE,
    OUTPUT_CONFLICT_SKIP,
    tk,
)
from auto_reframe_core.gui import (
    AutoReframeGUI,
    OUTPUT_CONFLICT_DELETE,
    build_job_confirmation_message,
    build_output_conflict_message,
)
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.video_utils import h264, h265


class JobConfirmationMessageTests(unittest.TestCase):
    def test_start_confirmation_uses_scrollable_dialog_on_main_thread(self):
        class StatusVariable:
            value = None

            def set(self, value):
                self.value = value

        for platform_name in ("win32", "darwin"):
            with self.subTest(platform=platform_name):
                app = AutoReframeGUI.__new__(AutoReframeGUI)
                app.running = False
                app.scan_busy = app.update_busy = app.preflight_busy = False
                app.root = SimpleNamespace(after=Mock())
                app.event_queue = queue.Queue()
                app._set_footer_buttons = Mock()
                app.status_var = StatusVariable()
                app._build_config = lambda: ("compress", CompressConfig())

                with (
                    patch("auto_reframe_core.gui.sys.platform", platform_name),
                    patch(
                        "auto_reframe_core.output_plans.prepare_video_batch",
                        return_value=SimpleNamespace(destinations=(), finals=()),
                    ),
                    patch(
                        "auto_reframe_core.gui_job_view.JobConfirmationDialog.ask",
                        return_value=False,
                    ) as confirmation,
                ):
                    app.start_job()
                    app.worker.join(5)
                    self.assertFalse(app.worker.is_alive())
                    confirmation.assert_not_called()
                    app._drain_events()

                confirmation.assert_called_once()
                self.assertEqual(
                    confirmation.call_args.args[0],
                    app.root,
                )
                self.assertEqual(app.status_var.value, "已取消開始處理")

    def test_output_conflict_message_explains_scope_and_all_actions(self):
        message = build_output_conflict_message(
            [
                Path("output/4x5_FHD_h265"),
                Path("output/4x5_HD_h265"),
            ]
        )

        self.assertIn("本次規劃中已存在的輸出檔案", message)
        self.assertIn("其他影片與檔案不受影響", message)
        self.assertIn("略過既有檔", message)
        self.assertIn("覆寫同名檔", message)
        self.assertIn("刪除本次輸出檔", message)
        self.assertIn("取消", message)

    def test_reframe_lists_every_target_normalized_text_and_watermark(self):
        config = ReframeConfig(
            final_ratio=(9, 16),
            targets=[
                {"ratio": (4, 5), "resolution": "source", "vcodec": h265},
                {"ratio": (1, 1), "resolution": "1080p", "vcodec": h264},
            ],
            top_text_override="標題\r\n第二行\n",
            bottom_text_override="說明\n",
            watermark_enabled=True,
            watermark_file="watermark/logo.png",
        )

        message = build_job_confirmation_message("reframe", config)

        self.assertIn("浮水印：已套用（logo.png）", message)
        self.assertIn("1. 裁切 4:5 → 畫布 9:16", message)
        self.assertIn("2. 裁切 1:1 → 畫布 9:16", message)
        self.assertIn("偵測到的上方文字：\n標題\n第二行", message)
        self.assertIn("偵測到的下方文字：\n說明", message)

    def test_compress_lists_every_target_and_reports_no_watermark(self):
        config = CompressConfig(
            targets=[
                {"resolution": "1080p", "vcodec": h264},
                {"resolution": "720p", "vcodec": h265},
            ]
        )

        message = build_job_confirmation_message("compress", config)

        self.assertIn("浮水印：未套用", message)
        self.assertIn("1. Full HD 1080p／H.264 / AVC", message)
        self.assertIn("2. HD 720p／H.265 / HEVC", message)
        self.assertNotIn("偵測到的上方文字", message)

    def test_job_confirmation_repeats_selected_existing_output_action(self):
        config = CompressConfig()

        message = build_job_confirmation_message(
            "compress",
            config,
            OUTPUT_CONFLICT_DELETE,
        )

        self.assertIn("既有輸出：刪除列出的本次輸出檔後重做", message)


class ConfirmationDialogLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display is required: {error}")
        cls.scaling = cls.root.tk.call("tk", "scaling")

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def tearDown(self):
        for child in self.root.winfo_children():
            child.destroy()
        self.root.tk.call("tk", "scaling", self.scaling)

    def assert_actions_visible(self, dialog, screen_width, screen_height):
        self.root.update()
        window = dialog.window
        self.assertGreaterEqual(window.winfo_rootx(), 0)
        self.assertGreaterEqual(window.winfo_rooty(), 0)
        self.assertLessEqual(window.winfo_rootx() + window.winfo_width(), screen_width)
        self.assertLessEqual(window.winfo_rooty() + window.winfo_height(), screen_height)
        for button in dialog.buttons.winfo_children():
            self.assertTrue(button.winfo_viewable())
            self.assertGreaterEqual(button.winfo_rootx(), window.winfo_rootx())
            self.assertLessEqual(
                button.winfo_rootx() + button.winfo_width(),
                window.winfo_rootx() + window.winfo_width(),
            )
            self.assertGreaterEqual(button.winfo_rooty(), window.winfo_rooty())
            self.assertLessEqual(
                button.winfo_rooty() + button.winfo_height(),
                window.winfo_rooty() + window.winfo_height(),
            )

    def test_long_summary_keeps_all_content_and_actions_in_small_windows(self):
        config = ReframeConfig(top_text_override="標題\n" * 500)
        config.prepared_batch = SimpleNamespace(
            destinations=tuple(Path("output") / ("long-folder-" * 20) / str(i) for i in range(80)),
        )
        config.video_files = (Path("source.mp4"),)
        summary = build_job_confirmation_message("reframe", config)
        for screen_width, screen_height, scaling in (
            (800, 600, 1.0), (1024, 768, 2.0), (800, 600, 2.67),
        ):
            with self.subTest(screen=(screen_width, screen_height), scaling=scaling):
                self.root.tk.call("tk", "scaling", scaling)
                self.root.geometry(f"640x480+{screen_width - 80}+{screen_height - 80}")
                self.root.update_idletasks()
                with (
                    patch.object(tk.Toplevel, "winfo_screenwidth", return_value=screen_width),
                    patch.object(tk.Toplevel, "winfo_screenheight", return_value=screen_height),
                ):
                    dialog = JobConfirmationDialog(self.root, summary)
                self.assert_actions_visible(dialog, screen_width, screen_height)
                self.assertEqual(dialog.message.get("1.0", "end-1c"), summary)
                self.assertEqual(str(dialog.message.cget("state")), "disabled")
                self.assertLess(dialog.message.yview()[1], 1.0)
                dialog.message.yview_moveto(1.0)
                self.root.update()
                self.assertAlmostEqual(dialog.message.yview()[1], 1.0)
                width, height = dialog.window.minsize()
                dialog.window.geometry(f"{width}x{height}")
                self.assert_actions_visible(dialog, screen_width, screen_height)
                dialog.buttons.winfo_children()[0].invoke()
                self.assertTrue(dialog.result)

    def test_long_conflict_paths_keep_every_choice_visible(self):
        self.root.geometry("640x480+720+520")
        self.root.update_idletasks()
        conflicts = [Path("output") / ("long-folder-" * 100) / f"{i}.mp4" for i in range(20)]
        for prefer_skip, expected, scaling in (
            (True, OUTPUT_CONFLICT_SKIP, 1.0), (False, OUTPUT_CONFLICT_OVERWRITE, 2.67),
        ):
            with self.subTest(prefer_skip=prefer_skip, scaling=scaling):
                self.root.tk.call("tk", "scaling", scaling)
                with (
                    patch.object(tk.Toplevel, "winfo_screenwidth", return_value=800),
                    patch.object(tk.Toplevel, "winfo_screenheight", return_value=600),
                ):
                    dialog = OutputConflictDialog(self.root, conflicts, prefer_skip)
                self.assert_actions_visible(dialog, 800, 600)
                self.assertEqual(dialog.message.get("1.0", "end-1c"), build_output_conflict_message(conflicts))
                self.assertLess(dialog.message.yview()[1], 1.0)
                width, height = dialog.window.minsize()
                dialog.window.geometry(f"{width}x{height}")
                self.assert_actions_visible(dialog, 800, 600)
                dialog.window.focus_force()
                dialog.window.event_generate("<Return>")
                self.root.update()
                self.assertEqual(dialog.result, expected)

    def test_enter_starts_and_escape_or_window_close_cancels(self):
        for action, expected in (("<Return>", True), ("<Escape>", False), ("close", False), ("button", False)):
            with self.subTest(action=action):
                dialog = JobConfirmationDialog(self.root, "summary")
                self.root.update()
                self.assertEqual(str(dialog.buttons.winfo_children()[0].cget("default")), "active")
                if action == "close":
                    dialog.window.tk.call(dialog.window.protocol("WM_DELETE_WINDOW"))
                elif action == "button":
                    dialog.buttons.winfo_children()[1].invoke()
                else:
                    dialog.window.focus_force()
                    dialog.window.event_generate(action)
                self.root.update()
                self.assertEqual(dialog.result, expected)
                self.assertFalse(dialog.window.winfo_exists())

        conflict = OutputConflictDialog(self.root, [Path("output.mp4")], True)
        conflict.buttons.winfo_children()[-1].invoke()
        self.assertEqual(conflict.result, OUTPUT_CONFLICT_CANCEL)

    def test_enter_on_focused_cancel_button_does_not_start(self):
        dialog = JobConfirmationDialog(self.root, "summary")
        self.root.update()
        dialog.buttons.winfo_children()[1].focus_force()
        dialog.window.event_generate("<Return>")
        self.root.update()
        self.assertFalse(dialog.result)
        self.assertFalse(dialog.window.winfo_exists())


if __name__ == "__main__":
    unittest.main()
