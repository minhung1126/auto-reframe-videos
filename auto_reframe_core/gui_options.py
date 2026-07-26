# -*- coding: utf-8 -*-
"""Pure option helpers shared by the desktop GUI and its tests."""

from pathlib import Path
from typing import List, Tuple


RESOLUTION_OPTIONS = (
    ("source", "原始解析度（不放大）"),
    ("4k", "4K（短邊上限 2160）"),
    ("2k", "2K / 1440p（短邊上限 1440）"),
    ("1080p", "Full HD 1080p"),
    ("720p", "HD 720p"),
    ("480p", "480p"),
    ("360p", "360p"),
)

CODEC_OPTIONS = (
    ("h264", "H.264 / AVC"),
    ("h265", "H.265 / HEVC"),
)

RATIO_OPTIONS = ("4:5", "1:1", "4:3", "16:9")

RESOLUTION_LABELS = dict(RESOLUTION_OPTIONS)
CODEC_LABELS = dict(CODEC_OPTIONS)
RESOLUTION_KEYS_BY_LABEL = {label: key for key, label in RESOLUTION_OPTIONS}
CODEC_KEYS_BY_LABEL = {label: key for key, label in CODEC_OPTIONS}


def parse_ratio(value: str) -> Tuple[int, int]:
    """Parse an editable W:H ratio without silently accepting malformed input."""
    parts = str(value).strip().replace("：", ":").split(":")
    if len(parts) != 2:
        raise ValueError("比例格式必須是「寬:高」，例如 4:5。")
    try:
        width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("比例的寬與高必須是整數。") from exc
    if width <= 0 or height <= 0:
        raise ValueError("比例的寬與高必須大於 0。")
    return width, height


def list_watermark_pngs(directory: Path) -> List[Path]:
    """Return all PNG files in deterministic, case-insensitive filename order."""
    if not directory.is_dir():
        return []
    return sorted(
        (
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() == ".png"
        ),
        key=lambda item: (item.name.casefold(), item.name),
    )
