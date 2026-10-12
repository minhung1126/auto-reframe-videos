"""GUI settings validation without importing Tk or probing media files."""

from copy import deepcopy
import math

from auto_reframe_core.config_store import ConfigStoreError, load_config
from auto_reframe_core.gui_options import CODEC_LABELS, RESOLUTION_LABELS
from auto_reframe_core.watermark import WATERMARK_POSITIONS, parse_watermark_opacity


GUI_MODES = ("reframe", "compress")
MODE_TITLES = {"reframe": "【裁切重製】", "compress": "【影片壓縮】"}
DEFAULT_WATERMARK_WIDTH_RATIOS = {"reframe": 0.15, "compress": 0.10}
DEFAULT_WATERMARK_POSITION = "bottom-center"
DEFAULT_WATERMARK_OPACITY = 0.8
DEFAULT_WATERMARK_MARGIN = 3


def _integer(value, field, *, minimum=0):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = int(value)
        if isinstance(value, float) and value != result:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigStoreError(f"{field} 必須是整數。") from exc
    if minimum is not None and result < minimum:
        raise ConfigStoreError(f"{field} 必須大於或等於 {minimum}。")
    return result


def _number(value, field):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigStoreError(f"{field} 必須是數字。") from exc
    if not math.isfinite(result):
        raise ConfigStoreError(f"{field} 必須是有限數字。")
    return result


def _boolean(value, field):
    if not isinstance(value, bool):
        raise ConfigStoreError(f"{field} 必須是 true 或 false。")
    return value


def _string(value, field):
    if not isinstance(value, str):
        raise ConfigStoreError(f"{field} 必須是文字。")
    return value


def _ratio(value, field):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigStoreError(f"{field} 必須是 [寬, 高]。")
    return tuple(_integer(part, field, minimum=1) for part in value)


def _validation_modes(validation_mode):
    if validation_mode is None:
        return GUI_MODES
    if validation_mode not in GUI_MODES:
        raise ConfigStoreError(f"設定驗證模式無效: {validation_mode!r}。")
    return (validation_mode,)


def normalize_target_sets(settings: dict, *, validation_mode=None) -> dict:
    """Validate and deduplicate the versioned document's per-mode targets."""
    raw_targets = settings.get("targets")
    if not isinstance(raw_targets, dict):
        raise ConfigStoreError("設定檔的 targets 必須是物件。")
    modes = _validation_modes(validation_mode)
    if "reframe" in modes:
        final_ratio = _ratio(settings.get("final_ratio"), "設定檔的 final_ratio")
    normalized = deepcopy(raw_targets) if validation_mode is not None else {}
    for mode in modes:
        normalized[mode] = []
        entries = raw_targets.get(mode)
        if not isinstance(entries, list) or not entries:
            raise ConfigStoreError(f"設定檔的 targets.{mode} 不可為空。")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigStoreError(f"targets.{mode} 的每個項目都必須是物件。")
            resolution = str(entry.get("resolution", "")).lower()
            codec = str(entry.get("vcodec", "")).lower()
            if resolution not in RESOLUTION_LABELS or codec not in CODEC_LABELS:
                raise ConfigStoreError(f"targets.{mode} 包含不支援的解析度或 codec。")
            target = {"resolution": resolution, "vcodec": codec}
            if mode == "reframe":
                ratio = _ratio(entry.get("ratio"), "重製 target 的 ratio")
                if ratio[0] * final_ratio[1] < final_ratio[0] * ratio[1]:
                    raise ConfigStoreError("中央影片比例不可比 final_ratio 更窄高。")
                target["ratio"] = ratio
            if target not in normalized[mode]:
                normalized[mode].append(target)
    return normalized


def normalize_watermark_settings(settings: dict, *, validation_mode=None) -> dict:
    """Use the same per-mode watermark rules for JSON and widget snapshots."""
    raw_watermarks = settings.get("watermarks", {})
    if not isinstance(raw_watermarks, dict):
        raise ConfigStoreError("設定檔的 watermarks 必須是物件。")
    normalized = deepcopy(raw_watermarks) if validation_mode is not None else {}
    for mode in _validation_modes(validation_mode):
        entry = raw_watermarks.get(mode, {})
        prefix = f"{MODE_TITLES[mode]}watermarks.{mode}"
        if not isinstance(entry, dict):
            raise ConfigStoreError(f"設定檔的 watermarks.{mode} 必須是物件。")
        enabled = _boolean(entry.get("enabled", False), f"{prefix}.enabled")
        file_value = _string(entry.get("file", ""), f"{prefix}.file").strip()
        position = _string(
            entry.get("position", DEFAULT_WATERMARK_POSITION), f"{prefix}.position"
        ).strip().lower()
        if position not in WATERMARK_POSITIONS:
            allowed = ", ".join(sorted(WATERMARK_POSITIONS))
            raise ConfigStoreError(f"{prefix}.position 無效: {position!r}。可用值: {allowed}")
        width_ratio = _number(
            entry.get("width_ratio", DEFAULT_WATERMARK_WIDTH_RATIOS[mode]),
            f"{prefix}.width_ratio",
        )
        if not 0.01 <= width_ratio <= 1.0:
            raise ConfigStoreError(f"{prefix}.width_ratio 必須介於 0.01 與 1.0。")
        try:
            raw_opacity = entry.get("opacity", DEFAULT_WATERMARK_OPACITY)
            if isinstance(raw_opacity, bool):
                raise ValueError
            opacity = parse_watermark_opacity(raw_opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigStoreError(
                f"{prefix}.opacity 必須介於 0.0 與 1.0（或 0% 與 100%）。"
            ) from exc
        margin = _integer(
            entry.get("margin", DEFAULT_WATERMARK_MARGIN),
            f"{prefix}.margin",
            minimum=None,
        )
        if not 0 <= margin <= 100:
            raise ConfigStoreError(f"{prefix}.margin 必須介於 0 與 100。")
        normalized[mode] = {
            "enabled": enabled,
            "file": file_value,
            "position": position,
            "width_ratio": width_ratio,
            "opacity": opacity,
            "margin": margin,
        }
    return normalized


def normalize_gui_settings(settings: dict, *, validation_mode=None) -> dict:
    """Validate a full document or the active job's mode and shared fields."""
    if not isinstance(settings, dict):
        raise ConfigStoreError("設定檔的 settings 必須是物件。")
    normalized = deepcopy(settings)
    mode = settings.get("mode")
    if mode not in GUI_MODES:
        raise ConfigStoreError(f"設定檔的 mode 無效: {mode!r}。")
    modes = _validation_modes(validation_mode)
    normalized["targets"] = normalize_target_sets(
        settings, validation_mode=validation_mode,
    )
    if "reframe" in modes:
        normalized["final_ratio"] = list(
            _ratio(settings.get("final_ratio"), "設定檔的 final_ratio")
        )
    normalized["watermarks"] = normalize_watermark_settings(
        settings, validation_mode=validation_mode,
    )
    output_mode = settings.get("output_mode", "source")
    if output_mode not in ("source", "specified"):
        raise ConfigStoreError(f"設定檔的 output_mode 無效: {output_mode!r}。")
    normalized["output_mode"] = output_mode
    normalized["output_folder"] = _string(
        settings.get("output_folder", ""), "輸出資料夾（output_folder）"
    ).strip()
    for key, default in (
        ("include_subfolders", False), ("skip_existing", True), ("debug", False)
    ):
        normalized[key] = _boolean(settings.get(key, default), key)
    string_fields = [("ffmpeg", "FFmpeg"), ("ffprobe", "FFprobe")]
    if "reframe" in modes:
        for key in ("top_text", "bottom_text"):
            normalized[key] = _string(settings.get(key, ""), key)
        string_fields.extend((("font_path", "字型檔"), ("font_color", "文字顏色")))
    for key, label in string_fields:
        value = _string(settings.get(key), f"{label}（{key}）").strip()
        if not value:
            if key == "font_path":
                raise ConfigStoreError("字型檔（font_path）不可為空。")
            value = "white" if key == "font_color" else key
        normalized[key] = value
    integer_fields = [("workers", "平行工作數", 0)]
    if "reframe" in modes:
        integer_fields.extend((
            ("top_font_size", "上方文字大小", 1),
            ("bottom_font_size", "下方文字大小", 1), ("text_margin", "文字邊距", 0),
        ))
    for key, label, minimum in integer_fields:
        normalized[key] = _integer(settings.get(key), f"{label}（{key}）", minimum=minimum)
    if "reframe" in modes:
        for key, label in (("top_spacing", "上方行距"), ("bottom_spacing", "下方行距")):
            value = _number(settings.get(key), f"{label}（{key}）")
            if value <= 0:
                raise ConfigStoreError(f"{label}（{key}）必須大於 0。")
            normalized[key] = value
    return normalized


def load_effective_settings(example_path, config_path):
    """Keep the complete committed example as the sole default source."""
    defaults = load_config(example_path)
    if defaults is None:
        raise ConfigStoreError(f"找不到預設設定: {example_path}")
    defaults = normalize_gui_settings(defaults)
    effective = deepcopy(defaults)
    saved = load_config(config_path)
    if saved:
        effective.update(saved)
    return defaults, normalize_gui_settings(effective)
