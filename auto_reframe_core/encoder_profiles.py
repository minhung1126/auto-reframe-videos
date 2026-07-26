# -*- coding: utf-8 -*-
"""Hardware encoder detection and FFmpeg encoder argument profiles."""

import subprocess
from typing import List, Optional, Tuple

from .platform_profile import PlatformProfile, current_platform


# Priority is preserved from the previous implementation:
# NVENC > AMF > QSV > VideoToolbox, except macOS prefers VideoToolbox first.
_HW_ENCODER_CANDIDATES = {
    "h265": [
        ("hevc_nvenc", "cuda"),
        ("hevc_amf", "d3d11va"),
        ("hevc_qsv", "qsv"),
        ("hevc_videotoolbox", "videotoolbox"),
    ],
    "h264": [
        ("h264_nvenc", "cuda"),
        ("h264_amf", "d3d11va"),
        ("h264_qsv", "qsv"),
        ("h264_videotoolbox", "videotoolbox"),
    ],
}

_SW_FALLBACK = {"h265": "libx265", "h264": "libx264"}


def _encoder_candidates(
    codec: str,
    profile: Optional[PlatformProfile] = None,
) -> List[Tuple[str, str]]:
    candidates = list(_HW_ENCODER_CANDIDATES[codec])
    p = profile or current_platform()
    if p.is_macos:
        vt = [c for c in candidates if c[1] == "videotoolbox"]
        other = [c for c in candidates if c[1] != "videotoolbox"]
        return vt + other
    return candidates


def _is_videotoolbox_encoder(encoder: str) -> bool:
    return encoder in ("hevc_videotoolbox", "h264_videotoolbox")


def _encoder_probe_cmd(ffmpeg_path: str, encoder: str) -> List[str]:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=256x256:d=1,format=yuv420p",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
    ]
    if _is_videotoolbox_encoder(encoder):
        cmd += ["-allow_sw", "0", "-realtime", "1"]
        if encoder == "hevc_videotoolbox":
            cmd += ["-profile:v", "main"]
    cmd += ["-f", "null", "-"]
    return cmd


def _detect_hw_encoder(
    ffmpeg_path: str,
    codec: str,
    profile: Optional[PlatformProfile] = None,
) -> Tuple[str, Optional[str]]:
    """Detect the first usable hardware encoder for h265 or h264."""
    try:
        res = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        encoders = res.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"呼叫 {ffmpeg_path} 失敗，請確認 FFmpeg 是否已安裝且路徑正確。"
        ) from exc
    if res.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 無法列出編碼器（exit code {res.returncode}）: {ffmpeg_path}"
        )

    label = codec.upper()
    for enc, hw in _encoder_candidates(codec, profile):
        if enc in encoders:
            try:
                test = subprocess.run(
                    _encoder_probe_cmd(ffmpeg_path, enc),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if test.returncode == 0:
                print(f"  [核心系統] 已啟用硬體加速編碼器 ({label}): {enc} ({hw})")
                return enc, hw

    sw = _SW_FALLBACK[codec]
    print(f"  [核心系統] 未發現可用硬體加速 ({label})，回退至軟體編碼 ({sw})")
    return sw, None


def detect_h265_hw_encoder(ffmpeg_path: str = "ffmpeg") -> Tuple[str, Optional[str]]:
    return _detect_hw_encoder(ffmpeg_path, "h265")


def detect_h264_hw_encoder(ffmpeg_path: str = "ffmpeg") -> Tuple[str, Optional[str]]:
    return _detect_hw_encoder(ffmpeg_path, "h264")


def detect_hwaccel_for_cmd(hwaccels: set) -> List[str]:
    """Return decode-side FFmpeg hwaccel flags for a compatible encoder set."""
    if len(hwaccels) != 1:
        return []
    hw = next(iter(hwaccels))
    if hw == "videotoolbox":
        # Keep VideoToolbox for encoding only; filter_complex uses CPU frames.
        return []
    return ["-hwaccel", hw]


def double_bitrate(vbr: str) -> str:
    import re

    m = re.fullmatch(r"([\d\.]+)([A-Za-z]*)", str(vbr))
    if not m:
        raise ValueError(f"無法解析 bitrate 字串: {vbr!r}")
    val = float(m.group(1)) * 2
    val_str = f"{int(val)}" if val.is_integer() else f"{val}"
    return f"{val_str}{m.group(2)}"


def build_encoder_args(encoder: str, vbr: str) -> List[str]:
    v = "v:0"
    if encoder in ("hevc_nvenc", "h264_nvenc"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-preset", "p4", "-rc", "vbr"]
    if encoder in ("hevc_amf", "h264_amf"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-quality", "balanced", "-rc", "vbr_latency"]
    if encoder in ("hevc_qsv", "h264_qsv"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-preset", "medium"]
    if _is_videotoolbox_encoder(encoder):
        args = [f"-c:{v}", encoder, f"-b:{v}", vbr, "-allow_sw", "0", "-realtime", "1"]
        if encoder == "hevc_videotoolbox":
            args += ["-profile:v", "main"]
        return args
    return [
        f"-c:{v}",
        encoder,
        "-preset",
        "medium",
        f"-b:{v}",
        vbr,
        f"-maxrate:{v}",
        double_bitrate(vbr),
        f"-bufsize:{v}",
        double_bitrate(vbr),
    ]


def build_output_args(encoder: str, vbr: str, vcodec: str, has_audio: bool, out_file) -> List[str]:
    args = build_encoder_args(encoder, vbr)
    args += ["-pix_fmt", "yuv420p"]
    if vcodec == "h265":
        args += ["-tag:v", "hvc1"]
    if has_audio:
        args += ["-c:a:0", "aac", "-b:a:0", "192k"]
    args += ["-f", "mp4", "-movflags", "+faststart", str(out_file)]
    return args
