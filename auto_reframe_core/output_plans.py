# -*- coding: utf-8 -*-
"""Output target planning shared by reframe and compress processors."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from video_utils import get_youtube_bitrate, resolve_short_side, resolution_label


@dataclass
class OutputPlan:
    active_maps: List[tuple]
    tmps: List[Path]
    finals: List[Path]

    @property
    def has_work(self) -> bool:
        return bool(self.active_maps)


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

    src_w, src_h = info["width"], info["height"]
    source_short = min(src_w, src_h)

    for target in config.targets:
        res_key = target["resolution"].lower()
        vcodec = target["vcodec"]

        final_short = resolve_short_side(res_key, source_short)
        scale_factor = min(final_short / source_short, 1.0) if source_short > 0 else 1.0

        out_w = int(src_w * scale_factor)
        out_h = int(src_h * scale_factor)
        out_w += out_w % 2
        out_h += out_h % 2

        label = f"COMPRESS_{resolution_label(final_short)}"
        watermark_suffix = "_wm" if getattr(config, "watermark_enabled", False) else ""
        suffix_name = f"{label}_{vcodec}{watermark_suffix}"
        sub_dir = out_dir / suffix_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        target_file = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

        if config.skip_existing and target_file.exists():
            continue

        tmp_file = target_file.with_name(target_file.name + ".tmp")
        bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
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

        out_w += out_w % 2
        out_h += out_h % 2

        label = resolution_label(final_short)
        watermark_suffix = "_wm" if getattr(config, "watermark_enabled", False) else ""
        suffix_name = f"{ratio_width}x{ratio_height}_{label}_{vcodec}{watermark_suffix}"
        sub_dir = out_dir / suffix_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        target_file = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

        if config.skip_existing and target_file.exists():
            continue

        tmp_file = target_file.with_name(target_file.name + ".tmp")
        bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
        active_maps.append((out_w, out_h, label, bitrate, tmp_file, vcodec))
        tmps.append(tmp_file)
        finals.append(target_file)

    return OutputPlan(active_maps, tmps, finals)
