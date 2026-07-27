# -*- coding: utf-8 -*-

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
from auto_reframe_core.video_utils import get_video_info


def _ffprobe_result(video_stream: dict) -> subprocess.CompletedProcess:
    data = {
        "streams": [video_stream],
        "format": {"duration": "1.25"},
    }
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=json.dumps(data),
        stderr="",
    )


class VideoInfoRotationTests(unittest.TestCase):
    @patch("auto_reframe_core.video_utils.subprocess.run")
    def test_unrotated_video_keeps_coded_dimensions(self, run):
        run.return_value = _ffprobe_result(
            {
                "codec_type": "video",
                "width": 320,
                "height": 180,
                "coded_width": 336,
                "coded_height": 192,
                "r_frame_rate": "30000/1001",
            }
        )

        info = get_video_info("ffprobe", Path("unrotated.mp4"))

        self.assertEqual((info["width"], info["height"]), (320, 180))
        self.assertEqual((info["source_width"], info["source_height"]), (320, 180))
        self.assertEqual((info["coded_width"], info["coded_height"]), (336, 192))
        self.assertEqual(info["rotation"], 0)

    @patch("auto_reframe_core.video_utils.subprocess.run")
    def test_display_matrix_quarter_turn_swaps_display_dimensions(self, run):
        run.return_value = _ffprobe_result(
            {
                "codec_type": "video",
                "width": 320,
                "height": 180,
                "r_frame_rate": "30/1",
                "side_data_list": [
                    {
                        "side_data_type": "Display Matrix",
                        "rotation": -90,
                    }
                ],
            }
        )

        info = get_video_info("ffprobe", Path("portrait.mp4"))

        self.assertEqual((info["width"], info["height"]), (180, 320))
        self.assertEqual((info["coded_width"], info["coded_height"]), (320, 180))
        self.assertEqual(info["rotation"], 270)
        dims = calculate_reframe_dimensions(
            info["width"], info["height"], (4, 5), (9, 16)
        )
        self.assertEqual(
            (dims["crop_w"], dims["crop_h"], dims["crop_x"], dims["crop_y"]),
            (180, 224, 0, 48),
        )

    @patch("auto_reframe_core.video_utils.subprocess.run")
    def test_rotate_tag_is_supported_case_insensitively(self, run):
        run.return_value = _ffprobe_result(
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "tags": {"ROTATE": "90"},
            }
        )

        info = get_video_info("ffprobe", Path("tagged.mov"))

        self.assertEqual((info["width"], info["height"]), (1080, 1920))
        self.assertEqual(info["rotation"], 90)

    @patch("auto_reframe_core.video_utils.subprocess.run")
    def test_display_matrix_takes_precedence_over_legacy_tag(self, run):
        run.return_value = _ffprobe_result(
            {
                "codec_type": "video",
                "width": 320,
                "height": 180,
                "r_frame_rate": "30/1",
                "tags": {"rotate": "90"},
                "side_data_list": [
                    {
                        "side_data_type": "Display Matrix",
                        "rotation": -180,
                    }
                ],
            }
        )

        info = get_video_info("ffprobe", Path("conflicting.mp4"))

        self.assertEqual((info["width"], info["height"]), (320, 180))
        self.assertEqual(info["rotation"], 180)

    @patch("auto_reframe_core.video_utils.subprocess.run")
    def test_display_matrix_precision_overrides_rounded_rotation_value(self, run):
        base_stream = {
            "codec_type": "video",
            "width": 320,
            "height": 180,
            "r_frame_rate": "30/1",
        }
        run.return_value = _ffprobe_result(
            {
                **base_stream,
                "side_data_list": [
                    {
                        "rotation": 89,
                        "displaymatrix": (
                            "\n00000000:           11      -65535           0"
                            "\n00000001:        65535          11           0"
                            "\n00000002:            0           0  1073741824\n"
                        ),
                    }
                ],
            }
        )

        near_quarter_turn = get_video_info("ffprobe", Path("near-90.mp4"))

        self.assertEqual(
            (near_quarter_turn["width"], near_quarter_turn["height"]),
            (180, 320),
        )
        self.assertEqual(near_quarter_turn["rotation"], 90)

        run.return_value = _ffprobe_result(
            {
                **base_stream,
                "side_data_list": [
                    {
                        "rotation": 90,
                        "displaymatrix": (
                            "\n00000000:        -1132      -65526           0"
                            "\n00000001:        65526       -1132           0"
                            "\n00000002:            0           0  1073741824\n"
                        ),
                    }
                ],
            }
        )

        general_rotation = get_video_info("ffprobe", Path("past-90.mp4"))

        self.assertEqual(
            (general_rotation["width"], general_rotation["height"]),
            (320, 180),
        )
        self.assertGreater(general_rotation["rotation"], 90.5)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg integration test requires ffmpeg and ffprobe",
)
class VideoInfoRotationFFmpegIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        help_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "-display_rotation" not in (help_result.stdout + help_result.stderr):
            raise unittest.SkipTest("FFmpeg does not support -display_rotation")

    def test_display_matrix_matches_ffmpeg_autorotated_frame_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            source = workdir / "source.mp4"
            rotated = workdir / "rotated.mp4"
            autorotated = workdir / "autorotated.mkv"

            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=1:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-display_rotation", "90", "-i", str(source),
                    "-map", "0", "-c", "copy", str(rotated),
                ],
                check=True,
            )

            info = get_video_info("ffprobe", rotated)
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(rotated), "-frames:v", "1",
                    "-c:v", "ffv1", str(autorotated),
                ],
                check=True,
            )
            autorotated_info = get_video_info("ffprobe", autorotated)

            self.assertEqual((info["source_width"], info["source_height"]), (320, 180))
            self.assertEqual((info["coded_width"], info["coded_height"]), (320, 180))
            self.assertEqual((info["width"], info["height"]), (180, 320))
            self.assertIn(info["rotation"], (90, 270))
            self.assertEqual(
                (autorotated_info["width"], autorotated_info["height"]),
                (info["width"], info["height"]),
            )

    def test_near_quarter_turn_dimensions_match_ffmpeg_autorotate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            source = workdir / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=1:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )

            for index, (angle, expected) in enumerate(
                ((89.99, (180, 320)), (90.99, (320, 180)))
            ):
                with self.subTest(angle=angle):
                    rotated = workdir / f"rotated-{index}.mp4"
                    autorotated = workdir / f"autorotated-{index}.mkv"
                    subprocess.run(
                        [
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-display_rotation", str(angle), "-i", str(source),
                            "-map", "0", "-c", "copy", str(rotated),
                        ],
                        check=True,
                    )
                    info = get_video_info("ffprobe", rotated)
                    subprocess.run(
                        [
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(rotated), "-frames:v", "1",
                            "-c:v", "ffv1", str(autorotated),
                        ],
                        check=True,
                    )
                    autorotated_info = get_video_info("ffprobe", autorotated)

                    self.assertEqual((info["width"], info["height"]), expected)
                    self.assertEqual(
                        (autorotated_info["width"], autorotated_info["height"]),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
