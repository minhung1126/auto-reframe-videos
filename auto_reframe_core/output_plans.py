# -*- coding: utf-8 -*-
"""Output target planning shared by reframe and compress processors."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from auto_reframe_core.video_utils import (
    get_youtube_bitrate,
    resolve_short_side,
    resolution_label,
)


@dataclass
class OutputPlan:
    active_maps: List[tuple]
    tmps: List[Path]
    finals: List[Path]

    @property
    def has_work(self) -> bool:
        return bool(self.active_maps)


def _encoder_compatible_dimension(value: int) -> int:
    """Round a dimension down to an encoder-compatible even value."""
    even_value = int(value) - int(value) % 2
    if even_value < 2:
        raise ValueError(f"輸出尺寸 {value} 無法在不放大的前提下對齊為有效偶數。")
    return even_value


def _register_output(
    seen_paths: dict,
    target_file: Path,
    effective_output: tuple,
) -> bool:
    """Deduplicate one path identity; reject paths that hide distinct outputs."""
    previous = seen_paths.get(target_file)
    if previous is None:
        seen_paths[target_file] = effective_output
        return True
    if previous == effective_output:
        return False
    raise ValueError(
        "多個 targets 解析成相同輸出路徑，但有效輸出尺寸不同："
        f"{target_file}（{previous[0]}x{previous[1]} 與 "
        f"{effective_output[0]}x{effective_output[1]}）。"
        "請移除其中一個 target，避免覆寫輸出。"
    )


def cleanup_temp_outputs(tmps: List[Path]) -> None:
    for tmp in tmps:
        if tmp.exists():
            tmp.unlink()


def promote_temp_outputs(tmps: List[Path], finals: List[Path]) -> None:
    for tmp, final in zip(tmps, finals):
        if tmp.exists():
            if final.exists():
                final.unlink()
            tmp.rename(final)


def build_compress_output_plan(config, out_dir: Path, file_path: Path, info: dict) -> OutputPlan:
    active_maps = []
    tmps = []
    finals = []
    seen_paths = {}

    src_w, src_h = info["width"], info["height"]
    source_short = min(src_w, src_h)

    for target in config.targets:
        res_key = target["resolution"].lower()
        vcodec = target["vcodec"]

        final_short = resolve_short_side(res_key, source_short)
        scale_factor = min(final_short / source_short, 1.0) if source_short > 0 else 1.0

        out_w = int(src_w * scale_factor)
        out_h = int(src_h * scale_factor)
        out_w = _encoder_compatible_dimension(out_w)
        out_h = _encoder_compatible_dimension(out_h)

        effective_short = min(out_w, out_h)
        label = f"COMPRESS_{resolution_label(effective_short)}"
        watermark_suffix = "_wm" if getattr(config, "watermark_enabled", False) else ""
        suffix_name = f"{label}_{vcodec}{watermark_suffix}"
        sub_dir = out_dir / suffix_name
        target_file = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

        effective_output = (out_w, out_h, vcodec, bool(watermark_suffix))
        if not _register_output(seen_paths, target_file, effective_output):
            continue

        if config.skip_existing and target_file.exists():
            continue

        sub_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = target_file.with_name(target_file.name + ".tmp")
        bitrate = get_youtube_bitrate(effective_short, info["fps"])
        active_maps.append((out_w, out_h, label, bitrate, tmp_file, vcodec))
        tmps.append(tmp_file)
        finals.append(target_file)

    return OutputPlan(active_maps, tmps, finals)


def build_reframe_output_plan(
    config,
    out_dir: Path,
    file_path: Path,
    info: dict,
    dims: dict,
    ratio_width: int,
    ratio_height: int,
    targets: list,
) -> OutputPlan:
    active_maps = []
    tmps = []
    finals = []
    seen_paths = {}

    for target in targets:
        res_key = target["resolution"].lower()
        vcodec = target["vcodec"]

        final_ratio_w, final_ratio_h = config.final_ratio
        source_short = min(dims["final_w"], dims["final_h"])
        final_short = resolve_short_side(res_key, source_short)

        if final_ratio_w <= final_ratio_h:
            out_w = final_short
            out_h = int(final_short * final_ratio_h / final_ratio_w)
        else:
            out_h = final_short
            out_w = int(final_short * final_ratio_w / final_ratio_h)

        out_w = _encoder_compatible_dimension(out_w)
        out_h = _encoder_compatible_dimension(out_h)

        effective_short = min(out_w, out_h)
        label = resolution_label(effective_short)
        watermark_suffix = "_wm" if getattr(config, "watermark_enabled", False) else ""
        suffix_name = f"{ratio_width}x{ratio_height}_{label}_{vcodec}{watermark_suffix}"
        sub_dir = out_dir / suffix_name
        target_file = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

        effective_output = (out_w, out_h, vcodec, bool(watermark_suffix))
        if not _register_output(seen_paths, target_file, effective_output):
            continue

        if config.skip_existing and target_file.exists():
            continue

        sub_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = target_file.with_name(target_file.name + ".tmp")
        bitrate = get_youtube_bitrate(effective_short, info["fps"])
        active_maps.append((out_w, out_h, label, bitrate, tmp_file, vcodec))
        tmps.append(tmp_file)
        finals.append(target_file)

    return OutputPlan(active_maps, tmps, finals)
