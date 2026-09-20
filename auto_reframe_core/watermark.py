# -*- coding: utf-8 -*-
"""Shared image-watermark configuration and FFmpeg filter helpers."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple


WATERMARK_POSITIONS = {
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
}

POSITION_ALIASES = {
    "left-center": "center-left",
    "right-center": "center-right",
    "middle-left": "center-left",
    "middle-right": "center-right",
    "middle-center": "center",
}


def parse_watermark_opacity(value: object) -> float:
    """Parse opacity supporting floats (0.0-1.0), percentages (0%-100%), and integer percentages (0-100)."""
    if isinstance(value, (int, float)):
        val = float(value)
        if 0.0 <= val <= 1.0:
            return round(val, 4)
        if val.is_integer() and 0 <= val <= 100:
            return round(val / 100.0, 4)
        raise ValueError("watermark_opacity 必須介於 0.0 與 1.0（或 0% 與 100%）。")

    text = str(value).strip()
    if not text:
        raise ValueError("浮水印透明度不可為空。")
    if text.endswith("%"):
        val = float(text[:-1].strip()) / 100.0
        if not (0.0 <= val <= 1.0):
            raise ValueError("watermark_opacity 必須介於 0% 與 100%。")
        return round(val, 4)

    val = float(text)
    if 0.0 <= val <= 1.0:
        return round(val, 4)
    if val.is_integer() and 0 <= val <= 100:
        return round(val / 100.0, 4)
    raise ValueError("watermark_opacity 必須介於 0.0 與 1.0（或 0% 與 100%）。")


@dataclass(frozen=True)
class WatermarkConfig:
    """Normalized settings for a single image watermark."""

    enabled: bool = False
    path: Optional[Path] = None
    position: str = "bottom-center"
    width_ratio: float = 0.07
    opacity: float = 0.8
    margin: int = 3


def build_watermark_config(
    *,
    enabled: bool,
    watermark_file: str,
    position: str,
    width_ratio: float,
    opacity: float = 0.8,
    margin: int,
    base_dir: Path,
) -> WatermarkConfig:
    """Validate user-facing values and resolve the watermark path."""
    raw_pos = str(position).strip().lower()
    normalized_position = POSITION_ALIASES.get(raw_pos, raw_pos)
    if normalized_position not in WATERMARK_POSITIONS:
        allowed = ", ".join(sorted(WATERMARK_POSITIONS))
        raise ValueError(f"watermark_position 無效: {position!r}。可用值: {allowed}")

    normalized_width = float(width_ratio)
    if not 0.01 <= normalized_width <= 1.0:
        raise ValueError("watermark_width_ratio 必須介於 0.01 與 1.0。")

    normalized_opacity = parse_watermark_opacity(opacity)

    normalized_margin = int(margin)
    if not 0 <= normalized_margin <= 100:
        raise ValueError("watermark_margin 必須介於 0 與 100。")

    raw_path = str(watermark_file).strip()
    resolved_path: Optional[Path] = None
    if raw_path:
        candidate = Path(raw_path).expanduser()
        resolved_path = candidate if candidate.is_absolute() else base_dir / candidate
        resolved_path = resolved_path.resolve()

    if enabled:
        if resolved_path is None:
            raise ValueError("啟用浮水印時必須選擇浮水印圖片。")
        if not resolved_path.is_file():
            raise ValueError(f"找不到浮水印圖片: {resolved_path}")

    return WatermarkConfig(
        enabled=bool(enabled),
        path=resolved_path,
        position=normalized_position,
        width_ratio=normalized_width,
        opacity=normalized_opacity,
        margin=normalized_margin,
    )


def watermark_overlay_xy(position: str, margin: int) -> Tuple[str, str]:
    """Return overlay x/y expressions for the selected anchor."""
    normalized_pos = POSITION_ALIASES.get(position, position)
    positions = {
        "top-left": (str(margin), str(margin)),
        "top-center": ("(main_w-overlay_w)/2", str(margin)),
        "top-right": (f"main_w-overlay_w-{margin}", str(margin)),
        "center-left": (str(margin), "(main_h-overlay_h)/2"),
        "center": ("(main_w-overlay_w)/2", "(main_h-overlay_h)/2"),
        "center-right": (f"main_w-overlay_w-{margin}", "(main_h-overlay_h)/2"),
        "bottom-left": (str(margin), f"main_h-overlay_h-{margin}"),
        "bottom-center": (
            "(main_w-overlay_w)/2",
            f"main_h-overlay_h-{margin}",
        ),
        "bottom-right": (
            f"main_w-overlay_w-{margin}",
            f"main_h-overlay_h-{margin}",
        ),
    }
    try:
        return positions[normalized_pos]
    except KeyError as exc:
        raise ValueError(f"不支援的浮水印位置: {position!r}") from exc


def lightroom_proportional_dimensions(
    out_width: int,
    out_height: int,
    watermark_width: int,
    watermark_height: int,
    proportional_ratio: float,
) -> Tuple[float, float]:
    """Return Lightroom-style proportional canvas dimensions before rounding."""
    scale = proportional_ratio * math.sqrt(
        (out_width * out_height) / (watermark_width * watermark_height)
    )
    return watermark_width * scale, watermark_height * scale


def watermark_width_expression(
    out_width: int,
    out_height: int,
    proportional_ratio: float,
) -> str:
    """Build the FFmpeg width expression using the watermark input aspect."""
    return (
        f"round(sqrt({out_width}*{out_height}*iw/ih)"
        f"*{proportional_ratio:g})"
    )


def watermark_inset_pixels(
    out_width: int,
    out_height: int,
    inset_percent: int,
) -> int:
    """Convert a Lightroom-style inset percentage to output pixels."""
    return max(
        0,
        int(round(math.sqrt(out_width * out_height) * inset_percent / 100.0)),
    )


def append_watermark_source_filter(
    filters: list,
    branch_count: int,
    config: WatermarkConfig,
    input_index: int = 1,
) -> list:
    """Prepare one reusable watermark branch for each output-resolution group."""
    if not config.enabled:
        return []
    if branch_count <= 0:
        raise ValueError("浮水印分支數必須大於 0。")

    if config.opacity < 1.0:
        source = (
            f"[{input_index}:v]format=rgba,"
            f"colorchannelmixer=aa={config.opacity:g}"
        )
    else:
        source = f"[{input_index}:v]format=rgba"
    labels = [f"[wm_src_{index}]" for index in range(branch_count)]
    if branch_count == 1:
        filters.append(f"{source}{labels[0]}")
    else:
        filters.append(f"{source},split={branch_count}{''.join(labels)}")
    return labels


def append_watermark_overlay_filter(
    filters: list,
    video_label: str,
    watermark_label: str,
    group_index: int,
    out_width: int,
    out_height: int,
    config: WatermarkConfig,
) -> str:
    """Scale and overlay a watermark on one finished resolution branch."""
    watermark_width = watermark_width_expression(
        out_width,
        out_height,
        config.width_ratio,
    )
    scaled_margin = watermark_inset_pixels(
        out_width,
        out_height,
        config.margin,
    )
    x_expr, y_expr = watermark_overlay_xy(config.position, scaled_margin)
    scaled_label = f"[wm_scaled_{group_index}]"
    output_label = f"[wm_out_{group_index}]"

    filters.append(
        f"{watermark_label}scale={watermark_width}:-2:"
        f"flags=lanczos{scaled_label};"
        f"{video_label}{scaled_label}overlay=x={x_expr}:y={y_expr}:"
        f"eof_action=repeat:shortest=0:repeatlast=1{output_label}"
    )
    return output_label
