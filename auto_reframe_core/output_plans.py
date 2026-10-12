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
from types import MappingProxyType
from stat import S_ISREG
from typing import List, Mapping

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


def _source_stamp(path: Path) -> tuple:
    try:
        stat = path.stat()
        if not S_ISREG(stat.st_mode):
            raise ValueError(f"來源影片不是檔案：{path}")
    except OSError as exc:
        raise ValueError(f"無法讀取來源影片：{path}") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _planning_signature(config, mode: str) -> tuple:
    targets = tuple((tuple(t.get("ratio", ())), t["resolution"].lower(), t["vcodec"])
                    for t in config.targets)
    resource_settings = tuple(getattr(config, name, None) for name in (
        "ffprobe_path", "font_path", "watermark_file", "watermark_position",
        "watermark_width_ratio", "watermark_opacity", "watermark_margin",
    ))
    return (mode, targets, tuple(getattr(config, "final_ratio", ())),
            getattr(config, "output_mode", "specified"),
            str(Path(config.output_dir).expanduser().resolve()),
            bool(getattr(config, "watermark_enabled", False)), resource_settings)


@dataclass(frozen=True)
class PreparedVideo:
    source: Path
    stamp: tuple
    info: Mapping
    plans: tuple

    def validate_source(self) -> None:
        if _source_stamp(self.source) != self.stamp:
            raise ValueError(f"來源影片在檢查後已變更，請重新開始處理：{self.source}")


@dataclass(frozen=True)
class PreparedVideoBatch:
    """One batch's validated metadata and output identities, without global caches."""

    mode: str
    signature: tuple
    videos: tuple
    entries: Mapping
    destinations: tuple
    finals: tuple
    final_keys: Mapping
    source_keys: frozenset

    def validate_sources(self) -> None:
        for entry in self.entries.values():
            entry.validate_source()

    def validate_video(self, file_path: Path) -> PreparedVideo:
        entry = self.entries.get(Path(file_path).resolve())
        if entry is None:
            raise ValueError(f"來源影片路徑在檢查後已變更：{file_path}")
        entry.validate_source()
        finals = tuple(final for _, plan in entry.plans for final in plan.finals)
        _validate_output_paths(self.videos, finals, self.final_keys, source_keys=self.source_keys)
        return entry

    def output_plan(self, file_path: Path, *, ratio=None, skip_existing=True) -> OutputPlan:
        entry = self.validate_video(file_path)
        snapshot = next(plan for key, plan in entry.plans if key == ratio)
        outputs = [(active, tmp, final)
                   for active, tmp, final in zip(snapshot.active_maps, snapshot.tmps, snapshot.finals)
                   if not (skip_existing and final.exists())]
        for _, _, final in outputs:
            final.parent.mkdir(parents=True, exist_ok=True)
        return OutputPlan([entry[0] for entry in outputs], [entry[1] for entry in outputs],
                          [entry[2] for entry in outputs])


def validate_prepared_batch(config, mode: str) -> PreparedVideoBatch:
    prepared = getattr(config, "prepared_batch", None)
    if not isinstance(prepared, PreparedVideoBatch):
        raise ValueError("尚未完成影片與輸出檢查。")
    videos = tuple(Path(path).resolve() for path in config.video_files)
    if prepared.signature != _planning_signature(config, mode) or prepared.videos != videos:
        raise ValueError("處理設定或影片清單在檢查後已變更，請重新開始處理。")
    _validate_batch_resources(config, mode)
    prepared.validate_sources()
    _validate_output_paths(prepared.videos, prepared.finals, prepared.final_keys)
    _check_output_writes(prepared.destinations, prepared.finals)
    return prepared


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
    errors = []
    for tmp in tmps:
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def promote_temp_outputs(tmps: List[Path], finals: List[Path], *, cancellation=None) -> None:
    """Replace a complete output group, restoring earlier finals if any replace fails."""
    if len(tmps) != len(finals):
        raise ValueError("暫存與完成輸出數量不一致。")
    canonical = {path: path.resolve() for path in (*tmps, *finals)}
    for tmp, final in zip(tmps, finals):
        if tmp.is_symlink() or not tmp.is_file() or tmp.stat().st_size == 0:
            raise ValueError(f"缺少完整暫存輸出：{tmp}")
        if final.is_symlink() or (final.exists() and not final.is_file()):
            raise ValueError(f"完成輸出路徑無效：{final}")
    backups = {}
    replaced = []
    keep_backups = set()
    try:
        # Hard links preserve large existing videos without another full copy on
        # native filesystems. Unsupported filesystems use a same-directory copy.
        for final in finals:
            if final.exists():
                fd, name = tempfile.mkstemp(prefix=".arv-promote-", suffix=".bak", dir=final.parent)
                os.close(fd)
                backup = Path(name)
                backups[final] = backup
                backup.unlink()
                try:
                    os.link(final, backup)
                except OSError:
                    shutil.copy2(final, backup)
        for tmp, final in zip(tmps, finals):
            if cancellation is not None and cancellation.cancelled:
                raise InterruptedError("輸出替換已取消。")
            # Check again immediately before replacing: an external application
            # could have changed output paths while FFmpeg was running.
            if (tmp.is_symlink() or final.is_symlink()
                    or tmp.resolve() != canonical[tmp] or final.resolve() != canonical[final]):
                raise ValueError(f"輸出路徑在替換期間已變更：{final}")
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise ValueError(f"缺少完整暫存輸出：{tmp}")
            tmp.replace(final)
            replaced.append(final)
    except BaseException as exc:
        rollback_errors = []
        for final in reversed(replaced):
            try:
                if final.resolve() != canonical[final]:
                    raise OSError(f"拒絕還原已變更的輸出路徑：{final}")
                if final in backups:
                    backups[final].replace(final)
                else:
                    final.unlink(missing_ok=True)
            except OSError as rollback_error:
                if final in backups:
                    keep_backups.add(backups[final])
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            raise RuntimeError(
                "輸出替換失敗且無法完整還原；原輸出備份保留於："
                + ", ".join(str(path) for path in keep_backups)
                + "；" + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for backup in backups.values():
            if backup not in keep_backups:
                backup.unlink(missing_ok=True)


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


def _validate_output_paths(videos, finals, expected_keys=None, *, source_keys=None):
    source_paths = ({_output_path_key(path) for path in videos}
                    if source_keys is None else source_keys)
    seen = set()
    for final in finals:
        tmp = final.with_name(final.name + ".tmp")
        key, tmp_key = _output_path_key(final), _output_path_key(tmp)
        if expected_keys is not None and key != expected_keys[final]:
            raise ValueError(f"輸出路徑在檢查後已變更：{final}")
        if key in source_paths or tmp_key in source_paths:
            raise ValueError(f"輸出不可覆寫原始影片：{final}")
        if key in seen or tmp_key in seen:
            raise ValueError(f"輸出路徑碰撞：{final}")
        if final.is_symlink() or tmp.is_symlink():
            raise ValueError(f"輸出不可使用符號連結：{final}")
        if tmp.exists() and not tmp.is_file():
            raise ValueError(f"暫存檔名已被資料夾占用：{tmp}")
        if final.exists() and not final.is_file():
            raise ValueError(f"輸出檔名已被資料夾占用：{final}")
        seen.update((key, tmp_key))


def _check_output_writes(destinations, finals):
    for directory in set(destinations) | {p.parent for p in finals}:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".arv-write-", dir=directory):
            pass


def _validate_batch_resources(config, mode):
    from auto_reframe_core.runtime_paths import resource_root
    from auto_reframe_core.target_specs import normalize_compress_target, normalize_reframe_target
    from auto_reframe_core.watermark import build_watermark_config
    if not config.targets:
        raise ValueError("targets 不可為空。")
    if mode == "reframe":
        final_ratio = config.final_ratio
        if (not isinstance(final_ratio, (tuple, list)) or len(final_ratio) != 2
                or int(final_ratio[0]) <= 0 or int(final_ratio[1]) <= 0):
            raise ValueError("ReframeConfig.final_ratio 必須是 2 個大於 0 的整數。")
        config.final_ratio = (int(final_ratio[0]), int(final_ratio[1]))
    for index, target in enumerate(config.targets, 1):
        if mode == "reframe":
            normalize_reframe_target(target, index, config.final_ratio)
        else:
            normalize_compress_target(target, index)
    root = resource_root()
    if mode == "reframe":
        font_path = root / config.font_path
        if not font_path.is_file():
            raise ValueError(f"找不到字型檔：{font_path}")
    watermark = build_watermark_config(
        enabled=config.watermark_enabled, watermark_file=config.watermark_file,
        position=config.watermark_position, width_ratio=config.watermark_width_ratio,
        opacity=config.watermark_opacity, margin=config.watermark_margin, base_dir=root,
    )
    if watermark.enabled:
        with watermark.path.open("rb") as source:
            source.read(1)


def prepare_video_batch(config, mode) -> PreparedVideoBatch:
    """Probe once and validate all identities before any worker or deletion starts."""
    from auto_reframe_core.video_utils import get_video_info
    if mode not in {"reframe", "compress"}:
        raise ValueError(f"不支援的輸出模式: {mode}")
    _validate_batch_resources(config, mode)
    videos = tuple(Path(p).resolve() for p in config.video_files)
    probe_config = replace(config, skip_existing=False)
    source_paths = {_output_path_key(p) for p in videos}
    seen = {}
    seen_tmps = {}
    destinations = set()
    finals = []
    entries = {}
    if mode == "reframe":
        from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
    for video in videos:
        stamp = _source_stamp(video)
        info = get_video_info(config.ffprobe_path, video)
        if not info:
            raise ValueError(f"無法讀取影片資訊：{video}")
        if _source_stamp(video) != stamp:
            raise ValueError(f"來源影片在讀取資訊期間已變更：{video}")
        root = output_root_for(config, video)
        destinations.add(root)
        plans = []
        if mode == "compress":
            plans.append((None, build_compress_output_plan(probe_config, root, video, info, prepare=False)))
        else:
            groups = {}
            for target in config.targets:
                groups.setdefault(target['ratio'], []).append(target)
            for ratio, targets in groups.items():
                dims = calculate_reframe_dimensions(info['width'], info['height'], ratio, config.final_ratio)
                plans.append((ratio, build_reframe_output_plan(probe_config, root, video, info, dims,
                                                              *ratio, targets, prepare=False)))
        entries[video] = PreparedVideo(video, stamp, MappingProxyType(dict(info)), tuple(plans))
        for _, plan in plans:
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
    destinations, finals = tuple(sorted(destinations)), tuple(finals)
    _check_output_writes(destinations, finals)
    prepared = PreparedVideoBatch(mode, _planning_signature(config, mode), videos,
                                  MappingProxyType(entries), destinations, finals,
                                  MappingProxyType({path: _output_path_key(path) for path in finals}),
                                  frozenset(source_paths))
    prepared.validate_sources()
    return prepared


def preflight_outputs(config, mode):
    """Compatibility interface for callers that only need destinations and finals."""
    prepared = prepare_video_batch(config, mode)
    return prepared.destinations, prepared.finals


def delete_planned_conflicts(conflicts, finals):
    allowed = {Path(p).resolve() for p in finals}
    for path in conflicts:
        path = Path(path)
        if path.resolve() not in allowed or not path.is_file():
            raise ValueError(f"拒絕刪除未規劃的輸出：{path}")
        path.unlink()
