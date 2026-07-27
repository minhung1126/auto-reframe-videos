# -*- coding: utf-8 -*-

from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from auto_reframe_core.reframe import ReframeConfig, VideoReframer
from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.encoder_profiles import (
    build_output_args,
    detect_h264_hw_encoder,
    detect_hwaccel_for_cmd,
)
from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.config_store import clear_config, load_config, save_config
from auto_reframe_core.gui import (
    CONFIG_EXAMPLE_PATH,
    CREDIT_SYMBOL,
    copy_text_to_clipboard,
    ensure_runtime_directories,
    normalize_target_sets,
)
from auto_reframe_core.output_plans import build_compress_output_plan, build_reframe_output_plan
from auto_reframe_core.platform_profile import PlatformProfile, resolve_workers
from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
from auto_reframe_core.text_layout import (
    TextLayoutConfig,
    append_fixed_reframe_text_filters,
    escape_drawtext_text,
    escape_filter_path,
)
from auto_reframe_core.gui_options import list_watermark_pngs, parse_ratio
from auto_reframe_core.watermark import WatermarkConfig, build_watermark_config
from auto_reframe_core.video_utils import get_video_info, h264, h265


class PlatformProfileTests(unittest.TestCase):
    def test_resolve_workers_preserves_platform_caps(self):
        self.assertEqual(resolve_workers(0, PlatformProfile("darwin", "posix", 16)), 4)
        self.assertEqual(resolve_workers(0, PlatformProfile("win32", "nt", 32)), 8)
        self.assertEqual(resolve_workers(99, PlatformProfile("linux", "posix", 4)), 4)

    def test_batch_runner_creates_fixed_input_and_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            config = SimpleNamespace(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                video_extensions={".mp4"},
                max_workers=1,
            )

            result = run_video_batch(config, lambda *_args: True, "測試")

            self.assertEqual(result, (0, []))
            self.assertTrue(input_dir.is_dir())
            self.assertTrue(output_dir.is_dir())


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

    @patch("auto_reframe_core.encoder_profiles.subprocess.run")
    def test_timed_out_encoder_probe_continues_to_the_next_backend(self, run):
        encoder_list = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="h264_nvenc\nh264_amf\n", stderr=""
        )
        amf_success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        run.side_effect = [
            encoder_list,
            subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=30),
            amf_success,
        ]

        encoder, hwaccel = detect_h264_hw_encoder("ffmpeg")

        self.assertEqual((encoder, hwaccel), ("h264_amf", "d3d11va"))

    @patch(
        "auto_reframe_core.encoder_profiles.subprocess.run",
        side_effect=FileNotFoundError("missing"),
    )
    def test_missing_ffmpeg_raises_a_catchable_error(self, _run):
        with self.assertRaises(RuntimeError):
            detect_h264_hw_encoder("missing-ffmpeg")


class ReframeLayoutGuardTests(unittest.TestCase):
    def test_drawtext_escaping_is_preserved(self):
        self.assertEqual(
            escape_drawtext_text(r"C:\A:B% 'quote'"),
            "C\\:\\\\A\\:B% '\\''quote'\\''",
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
        self.assertEqual(filter_complex.count("expansion=none"), 3)
        self.assertIn("y=420-20-ascent-1*line_h*1.08", filter_complex)
        self.assertIn("y=420-20-ascent-0*line_h*1.08", filter_complex)
        self.assertIn("y=1920-420+20+0*line_h*1.2", filter_complex)
        self.assertLess(filter_complex.index("[t_0_0]"), filter_complex.index("[b_0_0]"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_literal_percent_is_rendered_by_ffmpeg_without_expansion_error(self):
        font_path = Path(__file__).resolve().parents[1] / "fonts" / "NotoSerifTC.ttf"
        if not font_path.is_file():
            self.skipTest("Bundled test font is unavailable")

        layout = TextLayoutConfig(
            font_path=escape_filter_path(font_path),
            font_color="white",
            text_margin=0,
            top_font_size=640,
            bottom_font_size=640,
            top_line_spacing_ratio=1.0,
            bottom_line_spacing_ratio=1.0,
            top_text="100%",
            bottom_text="",
        )
        filter_complex, output_label = append_fixed_reframe_text_filters(
            "[0:v]null[base]",
            "[base]",
            0,
            120,
            {
                "pad_top": 100,
                "pad_bottom": 20,
                "final_h": 120,
            },
            layout,
        )
        result = subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x120:r=1:d=1",
                "-filter_complex",
                filter_complex,
                "-map",
                output_label,
                "-frames:v",
                "1",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")

        self.assertEqual(result.returncode, 0, stderr)
        self.assertNotIn("Stray %", stderr)
        self.assertEqual(len(result.stdout), 320 * 120)
        self.assertGreater(
            max(result.stdout),
            min(result.stdout),
            "drawtext produced a uniform blank frame",
        )


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

    def test_compress_output_plan_aligns_odd_source_down_without_upscaling(self):
        config = CompressConfig(
            output_dir="output",
            targets=[{"resolution": "source", "vcodec": h265}],
        )
        plan = build_compress_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"width": 1919, "height": 1079, "fps": 30.0},
        )

        out_w, out_h = plan.active_maps[0][:2]
        self.assertEqual((out_w, out_h), (1918, 1078))
        self.assertLessEqual(out_w, 1919)
        self.assertLessEqual(out_h, 1079)

    def test_compress_output_plan_deduplicates_equivalent_resolved_targets(self):
        config = CompressConfig(
            output_dir="output",
            targets=[
                {"resolution": "source", "vcodec": h265},
                {"resolution": "4k", "vcodec": h265},
            ],
        )
        plan = build_compress_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"width": 1920, "height": 1080, "fps": 30.0},
        )

        self.assertEqual(len(plan.active_maps), 1)
        self.assertEqual(len(plan.tmps), 1)
        self.assertEqual(len(plan.finals), 1)
        self.assertEqual(plan.active_maps[0][:2], (1920, 1080))

    def test_compress_output_plan_deduplicates_same_pixels_across_labels(self):
        config = CompressConfig(
            output_dir="output",
            targets=[
                {"resolution": "source", "vcodec": h265},
                {"resolution": "360p", "vcodec": h265},
            ],
        )
        plan = build_compress_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"width": 361, "height": 361, "fps": 30.0},
        )

        self.assertEqual(len(plan.active_maps), 1)
        self.assertEqual(plan.active_maps[0][:2], (360, 360))
        self.assertIn("COMPRESS_360P_h265", str(plan.finals[0]))

    def test_compress_output_plan_rejects_distinct_outputs_with_same_path(self):
        config = CompressConfig(
            output_dir="output",
            targets=[
                {"resolution": "source", "vcodec": h265},
                {"resolution": "720p", "vcodec": h265},
            ],
        )

        with self.assertRaisesRegex(ValueError, "相同輸出路徑"):
            build_compress_output_plan(
                config,
                Path("output"),
                Path("input/demo.mp4"),
                {"width": 1919, "height": 1079, "fps": 30.0},
            )

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

    def test_watermark_uses_distinct_output_identity(self):
        config = CompressConfig(
            output_dir="output",
            targets=[{"resolution": "1080p", "vcodec": h265}],
            watermark_enabled=True,
            watermark_file="watermark/logo.png",
        )
        plan = build_compress_output_plan(
            config,
            Path("output"),
            Path("input/demo.mp4"),
            {"width": 1920, "height": 1080, "fps": 30.0},
        )

        self.assertEqual(
            plan.finals[0],
            Path("output/COMPRESS_FHD_h265_wm/demo_COMPRESS_FHD_h265_wm.mp4"),
        )


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

    def test_watermark_is_applied_after_scale_and_before_codec_split(self):
        compressor = object.__new__(VideoCompressor)
        compressor.config = CompressConfig()
        compressor.config.ffmpeg_path = "ffmpeg"
        compressor.h265_encoder = "libx265"
        compressor.h265_hwaccel = None
        compressor.h264_encoder = "libx264"
        compressor.h264_hwaccel = None
        compressor.watermark = WatermarkConfig(
            enabled=True,
            path=Path("C:/含 空白/logo.png"),
            position="bottom-center",
            width_ratio=0.18,
            opacity=0.75,
            margin=48,
        )

        outputs = [
            (1920, 1080, "COMPRESS_FHD", "12M", Path("h264.tmp"), h264),
            (1920, 1080, "COMPRESS_FHD", "12M", Path("h265.tmp"), h265),
        ]
        info = {"width": 3840, "height": 2160, "has_audio": True}
        cmd = compressor.build_ffmpeg_split_command(Path("in.mp4"), outputs, info)
        filter_complex = cmd[cmd.index("-filter_complex") + 1]

        self.assertEqual(cmd.count("-i"), 2)
        self.assertIn(str(compressor.watermark.path), cmd)
        self.assertNotIn(str(compressor.watermark.path), filter_complex)
        self.assertEqual(filter_complex.count("overlay="), 1)
        self.assertIn("scale=1920:1080", filter_complex)
        self.assertIn("scale=346:-2:flags=lanczos", filter_complex)
        self.assertIn("x=(main_w-overlay_w)/2", filter_complex)
        self.assertIn("y=main_h-overlay_h-27", filter_complex)
        self.assertIn("shortest=0", filter_complex)
        self.assertLess(filter_complex.index("overlay="), filter_complex.index("split=2[out_0][out_1]"))
        self.assertEqual(cmd.count("0:a:0"), 2)

    @patch("auto_reframe_core.compress.run_ffmpeg_with_progress")
    @patch("auto_reframe_core.compress.get_video_info")
    def test_runtime_fallback_keeps_hw_encode_before_software(
        self, get_info, run_progress
    ):
        get_info.return_value = {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration": 1.0,
            "has_audio": False,
        }
        run_progress.side_effect = [
            (1, ["hardware decode failed"]),
            (1, ["hardware encode failed"]),
            (0, []),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            compressor = object.__new__(VideoCompressor)
            compressor.script_dir = Path(tmp)
            compressor.config = CompressConfig(
                output_dir=str(Path(tmp) / "output"),
                targets=[{"resolution": "1080p", "vcodec": h265}],
                skip_existing=False,
            )
            compressor.h265_encoder = "hevc_nvenc"
            compressor.h265_hwaccel = "cuda"
            compressor.h264_encoder = "h264_nvenc"
            compressor.h264_hwaccel = "cuda"
            compressor.watermark = WatermarkConfig()
            positions = Queue()
            positions.put(0)

            success = compressor.process_single_video(
                (1, 1, Path(tmp) / "input.mp4"), positions
            )

        self.assertTrue(success)
        commands = [call.args[0] for call in run_progress.call_args_list]
        self.assertIn("-hwaccel", commands[0])
        self.assertIn("hevc_nvenc", commands[0])
        self.assertNotIn("-hwaccel", commands[1])
        self.assertIn("hevc_nvenc", commands[1])
        self.assertNotIn("-hwaccel", commands[2])
        self.assertIn("libx265", commands[2])


class WatermarkAndGuiOptionTests(unittest.TestCase):
    def test_credit_symbol_copy_helper_uses_tk_clipboard(self):
        class FakeRoot:
            def __init__(self):
                self.clipboard = None
                self.updated = False

            def clipboard_clear(self):
                self.clipboard = ""

            def clipboard_append(self, value):
                self.clipboard += value

            def update_idletasks(self):
                self.updated = True

        root = FakeRoot()

        copy_text_to_clipboard(root, CREDIT_SYMBOL)

        self.assertEqual(CREDIT_SYMBOL, "©")
        self.assertEqual(root.clipboard, "©")
        self.assertTrue(root.updated)

    def test_watermark_png_scan_is_case_insensitive_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "z.PNG").write_bytes(b"")
            (directory / "A.png").write_bytes(b"")
            (directory / "ignore.jpg").write_bytes(b"")

            self.assertEqual(
                [item.name for item in list_watermark_pngs(directory)],
                ["A.png", "z.PNG"],
            )

    def test_ratio_parser_accepts_ascii_and_full_width_colon(self):
        self.assertEqual(parse_ratio("4:5"), (4, 5))
        self.assertEqual(parse_ratio(" 1：1 "), (1, 1))
        with self.assertRaises(ValueError):
            parse_ratio("9/16")

    def test_enabled_watermark_requires_an_existing_png_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with self.assertRaises(ValueError):
                build_watermark_config(
                    enabled=True,
                    watermark_file="watermark/missing.png",
                    position="bottom-center",
                    width_ratio=0.18,
                    opacity=0.75,
                    margin=48,
                    base_dir=base_dir,
                )

    def test_config_example_is_the_valid_default_source(self):
        settings = load_config(CONFIG_EXAMPLE_PATH)

        self.assertIsNotNone(settings)
        self.assertEqual(settings["watermark_file"], "")
        self.assertEqual(settings["watermark_position"], "bottom-center")
        self.assertEqual(settings["watermark_width_ratio"], 0.32)
        self.assertEqual(
            normalize_target_sets(settings)["reframe"][0]["ratio"],
            (4, 5),
        )

    def test_config_round_trip_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            settings = {
                "mode": "compress",
                "label": "繁體中文",
                "targets": {"compress": []},
            }

            save_config(config_path, settings)
            self.assertEqual(load_config(config_path), settings)
            clear_config(config_path)
            self.assertFalse(config_path.exists())

    def test_runtime_directory_helper_creates_every_fixed_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("input", "output", "watermark")]

            ensure_runtime_directories(paths)

            self.assertTrue(all(path.is_dir() for path in paths))


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg integration test requires ffmpeg and ffprobe",
)
class WatermarkFFmpegIntegrationTests(unittest.TestCase):
    def test_single_frame_png_is_repeated_for_the_full_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            input_file = workdir / "input.mp4"
            watermark_file = workdir / "logo.png"
            output_file = workdir / "output.mp4.tmp"

            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=0.8",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(input_file),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=white@0.8:s=64x32",
                    "-frames:v", "1", str(watermark_file),
                ],
                check=True,
            )

            compressor = object.__new__(VideoCompressor)
            compressor.config = CompressConfig()
            compressor.config.ffmpeg_path = "ffmpeg"
            compressor.h265_encoder = "libx265"
            compressor.h265_hwaccel = None
            compressor.h264_encoder = "libx264"
            compressor.h264_hwaccel = None
            compressor.watermark = WatermarkConfig(
                enabled=True,
                path=watermark_file,
                position="bottom-center",
                width_ratio=0.18,
                opacity=0.75,
                margin=48,
            )
            info = get_video_info("ffprobe", input_file)
            outputs = [(320, 180, "SOURCE", "1M", output_file, h264)]
            cmd = compressor.build_ffmpeg_split_command(input_file, outputs, info)

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-count_frames",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames:format=duration",
                    "-of", "json", str(output_file),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(probe.stdout)
            self.assertGreaterEqual(int(data["streams"][0]["nb_read_frames"]), 20)
            self.assertGreater(float(data["format"]["duration"]), 0.6)


if __name__ == "__main__":
    unittest.main()
