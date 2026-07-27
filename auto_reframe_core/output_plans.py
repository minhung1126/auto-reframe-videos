# -*- coding: utf-8 -*-
"""Output target planning shared by reframe and compress processors."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import List

from auto_reframe_core.video_utils import (
    RESOLUTION_MAP,
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


def _watermark_suffix(config) -> str:
    return "_wm" if getattr(config, "watermark_enabled", False) else ""


def compress_output_suffix(label: str, vcodec: str, watermark_suffix: str) -> str:
    return f"COMPRESS_{label}_{vcodec}{watermark_suffix}"


def reframe_output_suffix(
    ratio_width: int,
    ratio_height: int,
    label: str,
    vcodec: str,
    watermark_suffix: str,
) -> str:
    return f"{ratio_width}x{ratio_height}_{label}_{vcodec}{watermark_suffix}"


def _resolution_label_short_side(label: str) -> int | None:
    known = {
        "4K": 2160,
        "2K": 1440,
        "FHD": 1080,
        "HD": 720,
    }
    if label in known:
        return known[label]
    match = re.fullmatch(r"([1-9]\d*)P", label)
    return int(match.group(1)) if match else None


def _folder_matches_target(name: str, mode: str, target: dict, watermark_suffix: str) -> bool:
    vcodec = target["vcodec"]
    ending = f"_{vcodec}{watermark_suffix}"
    if not name.endswith(ending):
        return False

    if mode == "compress":
        prefix = "COMPRESS_"
    elif mode == "reframe":
        ratio_width, ratio_height = target["ratio"]
        prefix = f"{ratio_width}x{ratio_height}_"
    else:
        raise ValueError(f"不支援的輸出模式: {mode}")

    if not name.startswith(prefix):
        return False
    label = name[len(prefix):-len(ending)]
    short_side = _resolution_label_short_side(label)
    if short_side is None:
        return False

    resolution = target["resolution"].lower()
    if resolution == "source":
        return True
    return short_side <= RESOLUTION_MAP[resolution]


def find_target_output_conflicts(config, mode: str) -> List[Path]:
    """Find non-empty direct children that can be produced by the selected targets."""
    output_dir = Path(config.output_dir)
    if not output_dir.is_dir():
        return []

    watermark_suffix = _watermark_suffix(config)
    conflicts = []
    for child in sorted(output_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not any(
            _folder_matches_target(child.name, mode, target, watermark_suffix)
            for target in config.targets
        ):
            continue
        if child.is_dir() and not child.is_symlink():
            if next(child.iterdir(), None) is None:
                continue
        conflicts.append(child)
    return conflicts


def delete_target_output_conflicts(output_dir: Path, conflicts: List[Path]) -> None:
    """Delete only preflight paths that are direct children of the output root."""
    root = Path(output_dir).resolve()
    for conflict in conflicts:
        path = Path(conflict)
        if path.parent.resolve() != root:
            raise ValueError(f"拒絕刪除 output/ 以外的路徑: {path}")
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


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
        resolution_name = resolution_label(effective_short)
        label = f"COMPRESS_{resolution_name}"
        watermark_suffix = _watermark_suffix(config)
        suffix_name = compress_output_suffix(
            resolution_name,
            vcodec,
            watermark_suffix,
        )
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
        watermark_suffix = _watermark_suffix(config)
        suffix_name = reframe_output_suffix(
            ratio_width,
            ratio_height,
            label,
            vcodec,
            watermark_suffix,
        )
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
