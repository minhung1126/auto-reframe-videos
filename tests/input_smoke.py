# -*- coding: utf-8 -*-
"""Manual end-to-end smoke test using a short segment of a real input video."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.reframe import ReframeConfig, VideoReframer
from auto_reframe_core.video_utils import h264, h265


def run_checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, command))}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def probe(ffprobe_path: str, path: Path) -> dict:
    result = run_checked(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def assert_output(
    ffmpeg_path: str,
    ffprobe_path: str,
    output: Path,
    expected_codec: str,
    expected_size: tuple,
) -> dict:
    metadata = probe(ffprobe_path, output)
    stream = metadata["streams"][0]
    actual_size = (stream["width"], stream["height"])
    if stream["codec_name"] != expected_codec:
        raise AssertionError(
            f"{output.name}: expected codec {expected_codec}, got {stream['codec_name']}"
        )
    if actual_size != expected_size:
        raise AssertionError(
            f"{output.name}: expected {expected_size}, got {actual_size}"
        )
    if float(metadata["format"]["duration"]) < 1.0:
        raise AssertionError(f"{output.name}: output duration is unexpectedly short")

    run_checked(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    return {
        "file": output.name,
        "codec": stream["codec_name"],
        "size": f"{stream['width']}x{stream['height']}",
        "duration": round(float(metadata["format"]["duration"]), 3),
    }


def first_mp4(directory: Path) -> Path:
    outputs = sorted(directory.rglob("*.mp4"))
    if len(outputs) != 1:
        raise AssertionError(f"Expected one output under {directory}, got {len(outputs)}")
    return outputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input video not found: {source}")

    summaries = []
    with tempfile.TemporaryDirectory(prefix="auto-reframe-input-smoke-") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        input_dir.mkdir()
        sample = input_dir / "sample.mp4"
        watermark = root / "smoke-watermark.png"

        run_checked(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "0",
                "-t",
                str(args.seconds),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                str(sample),
            ]
        )
        run_checked(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=white@0.85:s=512x160",
                "-frames:v",
                "1",
                str(watermark),
            ]
        )

        cases = [
            (
                "compress_h264_clean",
                VideoCompressor,
                CompressConfig(
                    input_dir=str(input_dir),
                    output_dir=str(root / "compress_h264_clean"),
                    targets=[{"resolution": "1080p", "vcodec": h264}],
                    ffmpeg_path=args.ffmpeg,
                    ffprobe_path=args.ffprobe,
                    max_workers=1,
                    skip_existing=False,
                ),
                "h264",
                (1920, 1080),
            ),
            (
                "compress_h265_watermark",
                VideoCompressor,
                CompressConfig(
                    input_dir=str(input_dir),
                    output_dir=str(root / "compress_h265_watermark"),
                    targets=[{"resolution": "4k", "vcodec": h265}],
                    ffmpeg_path=args.ffmpeg,
                    ffprobe_path=args.ffprobe,
                    max_workers=1,
                    skip_existing=False,
                    watermark_enabled=True,
                    watermark_file=str(watermark),
                ),
                "hevc",
                (3840, 2160),
            ),
            (
                "reframe_h265_clean",
                VideoReframer,
                ReframeConfig(
                    input_dir=str(input_dir),
                    output_dir=str(root / "reframe_h265_clean"),
                    targets=[
                        {"ratio": (4, 5), "resolution": "source", "vcodec": h265}
                    ],
                    ffmpeg_path=args.ffmpeg,
                    ffprobe_path=args.ffprobe,
                    max_workers=1,
                    skip_existing=False,
                ),
                "hevc",
                (1728, 3072),
            ),
            (
                "reframe_h264_watermark",
                VideoReframer,
                ReframeConfig(
                    input_dir=str(input_dir),
                    output_dir=str(root / "reframe_h264_watermark"),
                    targets=[
                        {"ratio": (4, 5), "resolution": "1080p", "vcodec": h264}
                    ],
                    ffmpeg_path=args.ffmpeg,
                    ffprobe_path=args.ffprobe,
                    max_workers=1,
                    skip_existing=False,
                    watermark_enabled=True,
                    watermark_file=str(watermark),
                ),
                "h264",
                (1080, 1920),
            ),
        ]

        for name, processor_type, config, codec, size in cases:
            print(f"\n[SMOKE] {name}")
            success_count, failed_files = processor_type(config).run()
            if success_count != 1 or failed_files:
                raise AssertionError(
                    f"{name}: success={success_count}, failed={failed_files}"
                )
            output = first_mp4(Path(config.output_dir))
            summary = assert_output(
                args.ffmpeg, args.ffprobe, output, codec, size
            )
            summary["case"] = name
            summaries.append(summary)

    print("\nSMOKE_RESULT=" + json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
