# -*- coding: utf-8 -*-
"""Protected drawtext layout helpers for the fixed top/video/bottom design."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class TextLayoutConfig:
    font_path: str
    font_color: str
    text_margin: int
    top_font_size: int
    bottom_font_size: int
    top_line_spacing_ratio: float
    bottom_line_spacing_ratio: float
    top_text: str
    bottom_text: str


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("'", "'\\''")
        .replace(":", "\\:")
    )


def append_fixed_reframe_text_filters(
    seq: str,
    curr_lbl: str,
    group_index: int,
    out_h: int,
    dims: dict,
    layout: TextLayoutConfig,
) -> Tuple[str, str]:
    """
    Append drawtext filters for the project's fixed top/video/bottom layout.

    Invariants:
    - top_text is rendered in the top padding and grows upward from the video edge
    - bottom_text is rendered in the bottom padding and grows downward from the video edge
    - each text line receives its own drawtext filter; no newline is passed to FFmpeg
    - fix_bounds=true and the full FFmpeg variable name ascent are required
    - expansion=none keeps literal percent signs from being parsed as expressions
    """
    scale_rate = out_h / 1920.0
    border_color = layout.font_color

    if layout.top_text:
        fz = int(layout.top_font_size * scale_rate)
        bw = max(1, int(fz * 0.03))
        mar = int(layout.text_margin * scale_rate)
        ptop = int(dims["pad_top"] * (out_h / dims["final_h"]))
        lines = layout.top_text.splitlines()
        for line_index, line in enumerate(lines):
            text_esc = escape_drawtext_text(line)
            reverse_index = len(lines) - 1 - line_index
            y_pos = f"{ptop}-{mar}-ascent-{reverse_index}*line_h*{layout.top_line_spacing_ratio}"
            next_lbl = f"[t_{group_index}_{line_index}]"
            seq += (
                f";{curr_lbl}drawtext=fontfile='{layout.font_path}':text='{text_esc}':fontsize={fz}:"
                f"fontcolor={layout.font_color}:borderw={bw}:bordercolor={border_color}:"
                f"expansion=none:fix_bounds=true:x=(w-text_w)/2:y={y_pos}{next_lbl}"
            )
            curr_lbl = next_lbl

    if layout.bottom_text:
        fz = int(layout.bottom_font_size * scale_rate)
        bw = max(1, int(fz * 0.03))
        mar = int(layout.text_margin * scale_rate)
        pbtm = int(dims["pad_bottom"] * (out_h / dims["final_h"]))
        lines = layout.bottom_text.splitlines()
        for line_index, line in enumerate(lines):
            text_esc = escape_drawtext_text(line)
            y_pos = f"{out_h}-{pbtm}+{mar}+{line_index}*line_h*{layout.bottom_line_spacing_ratio}"
            next_lbl = f"[b_{group_index}_{line_index}]"
            seq += (
                f";{curr_lbl}drawtext=fontfile='{layout.font_path}':text='{text_esc}':fontsize={fz}:"
                f"fontcolor={layout.font_color}:borderw={bw}:bordercolor={border_color}:"
                f"expansion=none:fix_bounds=true:x=(w-text_w)/2:y={y_pos}{next_lbl}"
            )
            curr_lbl = next_lbl

    return seq, curr_lbl
