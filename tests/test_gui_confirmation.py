# -*- coding: utf-8 -*-
"""Tests for the pre-processing confirmation summary."""

import unittest

from auto_reframe_core.compress import CompressConfig
from auto_reframe_core.gui import build_job_confirmation_message
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.video_utils import h264, h265


class JobConfirmationMessageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
