# -*- coding: utf-8 -*-
"""Shared image-watermark configuration and FFmpeg filter helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


WATERMARK_POSITIONS = {
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
    "center",
}


@dataclass(frozen=True)
class WatermarkConfig:
    """Normalized settings for a single image watermark."""

    enabled: bool = False
    path: Optional[Path] = None
    position: str = "bottom-center"
    width_ratio: float = 0.32
    opacity: float = 0.85
    margin: int = 56


def build_watermark_config(
    *,
    enabled: bool,
    watermark_file: str,
    position: str,
    width_ratio: float,
    opacity: float,
    margin: int,
    base_dir: Path,
) -> WatermarkConfig:
    """Validate user-facing values and resolve the watermark path."""
    normalized_position = str(position).strip().lower()
    if normalized_position not in WATERMARK_POSITIONS:
        allowed = ", ".join(sorted(WATERMARK_POSITIONS))
        raise ValueError(f"watermark_position 無效: {position!r}。可用值: {allowed}")

    normalized_width = float(width_ratio)
    if not 0.01 <= normalized_width <= 1.0:
        raise ValueError("watermark_width_ratio 必須介於 0.01 與 1.0。")

    normalized_opacity = float(opacity)
    if not 0.0 <= normalized_opacity <= 1.0:
        raise ValueError("watermark_opacity 必須介於 0.0 與 1.0。")

    normalized_margin = int(margin)
    if normalized_margin < 0:
        raise ValueError("watermark_margin 不可小於 0。")

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
    positions = {
        "top-left": (str(margin), str(margin)),
        "top-right": (f"main_w-overlay_w-{margin}", str(margin)),
        "bottom-left": (str(margin), f"main_h-overlay_h-{margin}"),
        "bottom-center": (
            "(main_w-overlay_w)/2",
            f"main_h-overlay_h-{margin}",
        ),
        "bottom-right": (
            f"main_w-overlay_w-{margin}",
            f"main_h-overlay_h-{margin}",
        ),
        "center": ("(main_w-overlay_w)/2", "(main_h-overlay_h)/2"),
    }
    try:
        return positions[position]
    except KeyError as exc:
        raise ValueError(f"不支援的浮水印位置: {position!r}") from exc


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

    source = (
        f"[{input_index}:v]format=rgba,"
        f"colorchannelmixer=aa={config.opacity:g}"
    )
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
    watermark_width = max(2, int(round(out_width * config.width_ratio)))
    watermark_width += watermark_width % 2
    scaled_margin = max(0, int(round(config.margin * out_height / 1920.0)))
    x_expr, y_expr = watermark_overlay_xy(config.position, scaled_margin)
    scaled_label = f"[wm_scaled_{group_index}]"
    output_label = f"[wm_out_{group_index}]"

    filters.append(
        f"{watermark_label}scale={watermark_width}:-2:flags=lanczos{scaled_label};"
        f"{video_label}{scaled_label}overlay=x={x_expr}:y={y_expr}:"
        f"eof_action=repeat:shortest=0:repeatlast=1{output_label}"
    )
    return output_label
