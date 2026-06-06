# -*- coding: utf-8 -*-

from pathlib import Path
import unittest

from auto_reframe import ReframeConfig, VideoReframer
from auto_compress import CompressConfig, VideoCompressor
from auto_reframe_core.encoder_profiles import build_output_args, detect_hwaccel_for_cmd
from auto_reframe_core.output_plans import build_compress_output_plan, build_reframe_output_plan
from auto_reframe_core.platform_profile import PlatformProfile, resolve_workers
from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
from auto_reframe_core.text_layout import escape_drawtext_text, escape_filter_path
from video_utils import h264, h265


class PlatformProfileTests(unittest.TestCase):
    def test_resolve_workers_preserves_platform_caps(self):
        self.assertEqual(resolve_workers(0, PlatformProfile("darwin", "posix", 16)), 4)
        self.assertEqual(resolve_workers(0, PlatformProfile("win32", "nt", 32)), 8)
        self.assertEqual(resolve_workers(99, PlatformProfile("linux", "posix", 4)), 4)


class EncoderProfileTests(unittest.TestCase):
    def test_output_args_preserve_h265_mp4_options(self):
        args = build_output_args("libx265", "12M", "h265", True, "out.mp4")

        self.assertIn("-tag:v", args)
        self.assertIn("hvc1", args)
        self.assertIn("-pix_fmt", args)
        self.assertIn("yuv420p", args)
        self.assertIn("-c:a:0", args)
        self.assertIn("aac", args)
        self.assertEqual(args[-1], "out.mp4")

    def test_videotoolbox_is_not_used_for_decode_side_hwaccel(self):
        self.assertEqual(detect_hwaccel_for_cmd({"videotoolbox"}), [])
        self.assertEqual(detect_hwaccel_for_cmd({"cuda"}), ["-hwaccel", "cuda"])
        self.assertEqual(detect_hwaccel_for_cmd({"cuda", "qsv"}), [])


class ReframeLayoutGuardTests(unittest.TestCase):
    def test_drawtext_escaping_is_preserved(self):
        self.assertEqual(
            escape_drawtext_text(r"C:\A:B% 'quote'"),
            "C\\:\\\\A\\:B%% '\\''quote'\\''",
        )
        self.assertEqual(
            escape_filter_path(Path("C:/fonts/NotoSerifTC.ttf")),
            "C\\:/fonts/NotoSerifTC.ttf",
        )

    def _build_reframer(self):
        reframer = object.__new__(VideoReframer)
        reframer.script_dir = Path(__file__).resolve().parents[1]
        reframer.config = ReframeConfig(
            targets=[{"ratio": (4, 5), "resolution": "1080p", "vcodec": h265}],
            top_text_line_spacing_ratio=1.08,
            bottom_text_line_spacing_ratio=1.2,
        )
        reframer.config.top_text_content = "2026.03.22\n富邦悍將 vs 統一獅"
        reframer.config.bottom_text_content = "example.credit"
        reframer.h265_encoder = "libx265"
        reframer.h265_hwaccel = None
        reframer.h264_encoder = "libx264"
        reframer.h264_hwaccel = None
        return reframer

    def test_drawtext_layout_invariants_are_preserved(self):
        reframer = self._build_reframer()
        dims = {
            "crop_w": 1080,
            "crop_h": 1080,
            "crop_x": 0,
            "crop_y": 0,
            "pad_top": 420,
            "pad_bottom": 420,
            "final_w": 1080,
            "final_h": 1920,
        }
        outputs = [(1080, 1920, "FHD", "12M", Path("out.tmp"), h265)]
        info = {"has_audio": False}

        cmd = reframer.build_ffmpeg_split_command(Path("in.mp4"), dims, outputs, info)
        filter_complex = cmd[cmd.index("-filter_complex") + 1]

        self.assertEqual(filter_complex.count("drawtext="), 3)
        self.assertNotIn("\n", filter_complex)
        self.assertEqual(filter_complex.count("fix_bounds=true"), 3)
        self.assertIn("y=420-20-ascent-1*line_h*1.08", filter_complex)
        self.assertIn("y=420-20-ascent-0*line_h*1.08", filter_complex)
        self.assertIn("y=1920-420+20+0*line_h*1.2", filter_complex)
        self.assertLess(filter_complex.index("[t_0_0]"), filter_complex.index("[b_0_0]"))


class OutputPlanTests(unittest.TestCase):
    def test_compress_output_plan_preserves_suffixes(self):
        config = CompressConfig(
            output_dir="output",
            targets=[{"resolution": "1080p", "vcodec": h264}],
        )
        plan = build_compress_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"width": 3840, "height": 2160, "fps": 30.0},
        )

        self.assertEqual(plan.active_maps[0][:3], (1920, 1080, "COMPRESS_FHD"))
        self.assertEqual(plan.active_maps[0][5], h264)
        self.assertEqual(plan.finals[0], Path("output/COMPRESS_FHD_h264/demo_COMPRESS_FHD_h264.mp4"))

    def test_reframe_geometry_and_output_plan_preserve_dimensions(self):
        dims = calculate_reframe_dimensions(1920, 1080, (4, 5), (9, 16))
        self.assertEqual(
            dims,
            {
                "crop_w": 864,
                "crop_h": 1080,
                "crop_x": 528,
                "crop_y": 0,
                "pad_top": 228,
                "pad_bottom": 228,
                "final_w": 864,
                "final_h": 1536,
            },
        )

        config = ReframeConfig(
            output_dir="output",
            targets=[{"ratio": (4, 5), "resolution": "source", "vcodec": h265}],
        )
        plan = build_reframe_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"fps": 30.0},
            dims,
            4,
            5,
            config.targets,
        )

        self.assertEqual(plan.active_maps[0][:3], (864, 1536, "HD"))
        self.assertEqual(plan.active_maps[0][5], h265)
        self.assertEqual(plan.finals[0], Path("output/4x5_HD_h265/demo_4x5_HD_h265.mp4"))


class CompressCommandGuardTests(unittest.TestCase):
    def test_compress_command_preserves_direct_map_without_filter_complex(self):
        compressor = object.__new__(VideoCompressor)
        compressor.config = CompressConfig()
        compressor.config.ffmpeg_path = "ffmpeg"
        compressor.h265_encoder = "libx265"
        compressor.h265_hwaccel = None
        compressor.h264_encoder = "libx264"
        compressor.h264_hwaccel = None

        outputs = [(1920, 1080, "COMPRESS_FHD", "12M", Path("out.tmp"), h265)]
        info = {"width": 1920, "height": 1080, "has_audio": True}
        cmd = compressor.build_ffmpeg_split_command(Path("in.mp4"), outputs, info)

        self.assertNotIn("-filter_complex", cmd)
        self.assertIn("-map", cmd)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:v")
        self.assertIn("0:a:0", cmd)


if __name__ == "__main__":
    unittest.main()
