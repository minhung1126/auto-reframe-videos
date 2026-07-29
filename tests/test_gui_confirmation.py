# -*- coding: utf-8 -*-
"""Tests for the pre-processing confirmation summary."""

import unittest
from pathlib import Path
from unittest.mock import patch

from auto_reframe_core.compress import CompressConfig
from auto_reframe_core.gui import (
    AutoReframeGUI,
    OUTPUT_CONFLICT_DELETE,
    build_job_confirmation_message,
    build_output_conflict_message,
    messagebox,
)
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.video_utils import h264, h265


class JobConfirmationMessageTests(unittest.TestCase):
    def test_start_confirmation_defaults_to_yes_on_windows_and_macos(self):
        class StatusVariable:
            value = None

            def set(self, value):
                self.value = value

        for platform_name in ("win32", "darwin"):
            with self.subTest(platform=platform_name):
                app = AutoReframeGUI.__new__(AutoReframeGUI)
                app.running = False
                app.root = object()
                app.status_var = StatusVariable()
                app._build_config = lambda: ("compress", CompressConfig())

                with (
                    patch("auto_reframe_core.gui.sys.platform", platform_name),
                    patch(
                        "auto_reframe_core.gui.find_target_output_conflicts",
                        return_value=[],
                    ),
                    patch(
                        "auto_reframe_core.gui.messagebox.askyesno",
                        return_value=False,
                    ) as confirmation,
                ):
                    app.start_job()

                self.assertEqual(
                    confirmation.call_args.kwargs["default"],
                    messagebox.YES,
                )
                self.assertEqual(app.status_var.value, "已取消開始處理")

    def test_output_conflict_message_explains_scope_and_all_actions(self):
        message = build_output_conflict_message(
            [
                Path("output/4x5_FHD_h265"),
                Path("output/4x5_HD_h265"),
            ]
        )

        self.assertIn("與本次輸出目標相同", message)
        self.assertIn("output/ 內其他資料夾與檔案不受影響", message)
        self.assertIn("略過既有檔", message)
        self.assertIn("覆寫同名檔", message)
        self.assertIn("刪除目標資料夾", message)
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

        self.assertIn("既有輸出：刪除列出的目標資料夾後完整重做", message)


if __name__ == "__main__":
    unittest.main()
