# -*- coding: utf-8 -*-
"""Output target planning shared by reframe and compress processors."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import os
import tempfile
import sys
import unicodedata
from dataclasses import replace
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


def build_compress_output_plan(config, out_dir: Path, file_path: Path, info: dict, *, prepare=True) -> OutputPlan:
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

        if prepare:
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
    *, prepare=True,
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

        if prepare:
            sub_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = target_file.with_name(target_file.name + ".tmp")
        bitrate = get_youtube_bitrate(effective_short, info["fps"])
        active_maps.append((out_w, out_h, label, bitrate, tmp_file, vcodec))
        tmps.append(tmp_file)
        finals.append(target_file)

    return OutputPlan(active_maps, tmps, finals)


def output_root_for(config, file_path):
    if getattr(config, "output_mode", "specified") == "source":
        return Path(file_path).resolve().parent / "auto-reframe"
    return Path(config.output_dir).expanduser().resolve()


def _output_path_key(path):
    value = str(Path(path).resolve())
    # Reject ambiguous names conservatively on the usual case-insensitive native filesystems.
    if sys.platform in {"win32", "darwin"}:
        return unicodedata.normalize("NFC", value).casefold()
    return os.path.normcase(value)


def preflight_outputs(config, mode):
    """Resolve all target identities before starting any worker or deleting output."""
    from auto_reframe_core.video_utils import get_video_info
    videos = tuple(Path(p).resolve() for p in config.video_files)
    probe_config = replace(config, skip_existing=False)
    source_paths = {_output_path_key(p) for p in videos}
    seen = {}
    seen_tmps = {}
    destinations = set()
    finals = []
    if mode == "reframe":
        from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
    for video in videos:
        info = get_video_info(config.ffprobe_path, video)
        if not info:
            raise ValueError(f"無法讀取影片資訊：{video}")
        root = output_root_for(config, video)
        destinations.add(root)
        plans = []
        if mode == "compress":
            plans.append(build_compress_output_plan(probe_config, root, video, info, prepare=False))
        else:
            groups = {}
            for target in config.targets:
                groups.setdefault(target['ratio'], []).append(target)
            for ratio, targets in groups.items():
                dims = calculate_reframe_dimensions(info['width'], info['height'], ratio, config.final_ratio)
                plans.append(build_reframe_output_plan(probe_config, root, video, info, dims,
                                                       *ratio, targets, prepare=False))
        for plan in plans:
            for final in plan.finals:
                key = _output_path_key(final)
                tmp_key = _output_path_key(final.with_name(final.name + '.tmp'))
                if key in source_paths or tmp_key in source_paths:
                    raise ValueError(f"輸出不可覆寫原始影片：{final}")
                if key in seen or key in seen_tmps or tmp_key in seen or tmp_key in seen_tmps:
                    previous = seen.get(key) or seen_tmps.get(key) or seen.get(tmp_key) or seen_tmps.get(tmp_key)
                    raise ValueError(f"輸出路徑碰撞：{previous} 與 {video} → {final}")
                if final.is_symlink() or final.with_name(final.name + ".tmp").is_symlink():
                    raise ValueError(f"輸出不可使用符號連結：{final}")
                tmp = final.with_name(final.name + ".tmp")
                if tmp.exists() and not tmp.is_file():
                    raise ValueError(f"暫存檔名已被資料夾占用：{tmp}")
                if final.exists() and not final.is_file():
                    raise ValueError(f"輸出檔名已被資料夾占用：{final}")
                seen[key] = video
                seen_tmps[tmp_key] = video
                finals.append(final)
    # Test actual directory writes rather than only permission bits.
    for directory in destinations | {p.parent for p in finals}:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.arv-write-', dir=directory):
            pass
    return tuple(sorted(destinations)), tuple(finals)


def delete_planned_conflicts(conflicts, finals):
    allowed = {Path(p).resolve() for p in finals}
    for path in conflicts:
        path = Path(path)
        if path.resolve() not in allowed or not path.is_file():
            raise ValueError(f"拒絕刪除未規劃的輸出：{path}")
        path.unlink()
