"""Desktop settings presentation and interaction."""

from copy import deepcopy
from pathlib import Path
from auto_reframe_core.compress import CompressConfig
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.gui_options import (
    CODEC_LABELS,
    RESOLUTION_LABELS,
    WATERMARK_POSITION_KEYS_BY_LABEL,
    WATERMARK_POSITION_LABELS,
)
from auto_reframe_core.config_store import ConfigStoreError
from auto_reframe_core.gui_settings import (
    DEFAULT_WATERMARK_OPACITY,
    normalize_gui_settings,
    normalize_target_sets,
    normalize_watermark_settings,
)
from auto_reframe_core.runtime_paths import is_frozen, tool_path
import tkinter as tk
from tkinter import messagebox
from auto_reframe_core.gui_constants import CONFIG_PATH, MODE_LABELS, SCRIPT_DIR


class SettingsView:
    def _show_field_error(self, message):
        if hasattr(self, "field_error_var"):
            self.field_error_var.set(str(message))
        variables = {
            "output_subfolder": "output_subfolder_var",
            "output_folder": "output_folder_var",
            "workers": "workers_var",
            "font_path": "font_path_var",
            "font_color": "font_color_var",
            "top_font_size": "top_font_size_var",
            "bottom_font_size": "bottom_font_size_var",
            "text_margin": "text_margin_var",
            "top_spacing": "top_spacing_var",
            "bottom_spacing": "bottom_spacing_var",
        }
        variable = next(
            (
                getattr(self, name, None)
                for key, name in variables.items()
                if key in str(message)
            ),
            None,
        )
        if variable is None or not hasattr(self, "notebook"):
            return

        def visit(widget):
            try:
                if str(widget.cget("textvariable")) == str(variable):
                    cursor = widget
                    while cursor.master is not None:
                        if cursor.master.winfo_class() == "TNotebook":
                            cursor.master.select(cursor)
                        cursor = cursor.master
                    widget.focus_set()
                    if hasattr(widget, "selection_range"):
                        widget.selection_range(0, "end")
                    if hasattr(widget, "state"):
                        widget.state(["invalid"])
                        variable.trace_add(
                            "write", lambda *_: widget.state(["!invalid"])
                        )
                    return True
            except tk.TclError:
                pass
            return any(visit(child) for child in widget.winfo_children())

        visit(self.notebook)

    def _create_variables(self, settings):
        settings = normalize_gui_settings(settings)
        self.output_mode_var = tk.StringVar(value=settings.get("output_mode", "source"))
        self.output_folder_var = tk.StringVar(value=settings.get("output_folder", ""))
        self.output_subfolder_var = tk.StringVar(
            value=settings.get("output_subfolder", "")
        )
        self.recursive_var = tk.BooleanVar(
            value=settings.get("include_subfolders", False)
        )
        self.video_count_var = tk.StringVar(value="影片總數：0")
        mode = str(settings.get("mode", ""))
        if mode not in MODE_LABELS:
            raise ConfigStoreError(f"不支援的預設模式: {mode!r}")
        self.mode_var = tk.StringVar(value=MODE_LABELS[mode])
        first_ratio = self.targets["reframe"][0]["ratio"]
        self.ratio_var = tk.StringVar(value=f"{first_ratio[0]}:{first_ratio[1]}")
        self.resolution_vars = {
            target_mode: tk.StringVar(
                value=RESOLUTION_LABELS[self.targets[target_mode][0]["resolution"]]
            )
            for target_mode in MODE_LABELS
        }
        self.codec_vars = {
            target_mode: tk.StringVar(
                value=CODEC_LABELS[self.targets[target_mode][0]["vcodec"]]
            )
            for target_mode in MODE_LABELS
        }

        self.watermark_enabled_vars = {
            mode: tk.BooleanVar(value=bool(self.watermark_settings[mode]["enabled"]))
            for mode in MODE_LABELS
        }
        self.watermark_file_vars = {
            mode: tk.StringVar(value=str(self.watermark_settings[mode]["file"]))
            for mode in MODE_LABELS
        }
        self.watermark_position_vars = {
            mode: tk.StringVar(
                value=WATERMARK_POSITION_LABELS.get(
                    self.watermark_settings[mode]["position"],
                    WATERMARK_POSITION_LABELS["bottom-center"],
                )
            )
            for mode in MODE_LABELS
        }
        self.watermark_width_ratio_vars = {
            mode: tk.StringVar(
                value=f"{float(self.watermark_settings[mode]['width_ratio']):g}"
            )
            for mode in MODE_LABELS
        }
        self.watermark_opacity_vars = {
            mode: tk.StringVar(
                value=f"{float(self.watermark_settings[mode].get('opacity', DEFAULT_WATERMARK_OPACITY)):g}"
            )
            for mode in MODE_LABELS
        }
        self.watermark_margin_vars = {
            mode: tk.StringVar(value=str(int(self.watermark_settings[mode]["margin"])))
            for mode in MODE_LABELS
        }
        for mode in MODE_LABELS:
            self.watermark_position_vars[mode].trace_add(
                "write",
                lambda *_, m=mode: self._sync_watermark_position_var(m),
            )

        font_path = Path(str(settings["font_path"]))
        if not font_path.is_absolute():
            font_path = self.resource_dir / font_path
        self.font_path_var = tk.StringVar(value=str(font_path))
        self.font_color_var = tk.StringVar(value=str(settings["font_color"]))
        self.top_font_size_var = tk.StringVar(value=str(settings["top_font_size"]))
        self.bottom_font_size_var = tk.StringVar(
            value=str(settings["bottom_font_size"])
        )
        self.text_margin_var = tk.StringVar(value=str(settings["text_margin"]))
        self.top_spacing_var = tk.StringVar(value=str(settings["top_spacing"]))
        self.bottom_spacing_var = tk.StringVar(value=str(settings["bottom_spacing"]))

        self.ffmpeg_var = tk.StringVar(value=tool_path("ffmpeg", settings["ffmpeg"]))
        self.ffprobe_var = tk.StringVar(value=tool_path("ffprobe", settings["ffprobe"]))
        self.workers_var = tk.StringVar(value=str(settings["workers"]))
        self.skip_existing_var = tk.BooleanVar(value=bool(settings["skip_existing"]))
        self.debug_var = tk.BooleanVar(value=bool(settings["debug"]))
        self.status_var = tk.StringVar(value="準備就緒")
        self.progress_summary_var = tk.StringVar(value="尚未開始處理")
        self.update_status_var = tk.StringVar(value="尚未檢查更新")

    def _sync_watermark_position_var(self, mode: str) -> None:
        val = self.watermark_position_vars[mode].get()
        if val in WATERMARK_POSITION_LABELS:
            self.watermark_position_vars[mode].set(WATERMARK_POSITION_LABELS[val])

    def _build_config(self):
        settings = self._collect_settings(
            portable_paths=False, validation_mode=self._mode_key()
        )
        mode = settings["mode"]
        videos = tuple(self.video_files)
        if not videos:
            raise ValueError("請先加入待處理影片。")
        output_mode = settings["output_mode"]
        folder = settings["output_folder"]
        if output_mode == "specified" and not folder:
            raise ValueError("請選擇輸出資料夾。")
        watermark = settings["watermarks"][mode]
        watermark_path = self.watermark_paths.get(watermark["file"])
        if watermark["enabled"] and watermark_path is None:
            raise ValueError("已啟用浮水印，但 watermark/ 中沒有可用的 PNG。")

        common = dict(
            video_files=videos,
            output_mode=output_mode,
            output_dir=folder,
            output_subfolder=settings["output_subfolder"],
            targets=settings["targets"][mode],
            ffmpeg_path=self.ffmpeg_var.get().strip() or "ffmpeg",
            ffprobe_path=self.ffprobe_var.get().strip() or "ffprobe",
            skip_existing=settings["skip_existing"],
            max_workers=settings["workers"],
            debug=settings["debug"],
            watermark_enabled=watermark["enabled"],
            watermark_file=str(watermark_path) if watermark_path else "",
            watermark_position=watermark["position"],
            watermark_width_ratio=watermark["width_ratio"],
            watermark_opacity=watermark["opacity"],
            watermark_margin=watermark["margin"],
        )
        if mode == "compress":
            return mode, CompressConfig(**common)

        font_path = Path(settings["font_path"]).expanduser()
        if not font_path.is_absolute():
            font_path = self.resource_dir / font_path
        config = ReframeConfig(
            **common,
            final_ratio=tuple(settings["final_ratio"]),
            top_text_override=settings["top_text"],
            bottom_text_override=settings["bottom_text"],
            font_path=str(font_path),
            font_color=settings["font_color"],
            top_font_size=settings["top_font_size"],
            bottom_font_size=settings["bottom_font_size"],
            text_margin=settings["text_margin"],
            top_text_line_spacing_ratio=settings["top_spacing"],
            bottom_text_line_spacing_ratio=settings["bottom_spacing"],
        )
        return mode, config

    def _collect_settings(self, *, portable_paths=True, validation_mode=None) -> dict:
        """Collect GUI settings independently of the transient selected-file list."""
        watermarks = {}
        for mode in ("reframe", "compress"):
            pos_label = self.watermark_position_vars[mode].get().strip()
            watermarks[mode] = {
                "enabled": self.watermark_enabled_vars[mode].get(),
                "file": self.watermark_file_vars[mode].get(),
                "position": WATERMARK_POSITION_KEYS_BY_LABEL.get(pos_label, pos_label),
                "width_ratio": self.watermark_width_ratio_vars[mode].get(),
                "opacity": self.watermark_opacity_vars[mode].get(),
                "margin": self.watermark_margin_vars[mode].get(),
            }

        raw_settings = {
            "output_mode": self.output_mode_var.get(),
            "output_folder": self.output_folder_var.get(),
            "output_subfolder": self.output_subfolder_var.get(),
            "include_subfolders": self.recursive_var.get(),
            "mode": self._mode_key(),
            "targets": deepcopy(self.targets),
            "final_ratio": deepcopy(self.settings.get("final_ratio")),
            "watermarks": watermarks,
            "top_text": self.top_text.get("1.0", "end-1c"),
            "bottom_text": self.bottom_text.get("1.0", "end-1c"),
            "font_path": self.font_path_var.get(),
            "font_color": self.font_color_var.get(),
            "top_font_size": self.top_font_size_var.get(),
            "bottom_font_size": self.bottom_font_size_var.get(),
            "text_margin": self.text_margin_var.get(),
            "top_spacing": self.top_spacing_var.get(),
            "bottom_spacing": self.bottom_spacing_var.get(),
            "ffmpeg": "ffmpeg"
            if is_frozen()
            else self.ffmpeg_var.get().strip() or "ffmpeg",
            "ffprobe": "ffprobe"
            if is_frozen()
            else self.ffprobe_var.get().strip() or "ffprobe",
            "workers": self.workers_var.get(),
            "skip_existing": self.skip_existing_var.get(),
            "debug": self.debug_var.get(),
        }
        try:
            settings = normalize_gui_settings(
                raw_settings, validation_mode=validation_mode
            )
        except ConfigStoreError as exc:
            raise ValueError(str(exc)) from exc
        if portable_paths:
            font_path = Path(settings["font_path"]).expanduser()
            if (
                font_path.resolve()
                == (self.resource_dir / "fonts" / "NotoSerifTC.ttf").resolve()
            ):
                # macOS resource symlinks preserve the font's identity after app moves.
                settings["font_path"] = "fonts/NotoSerifTC.ttf"
            else:
                try:
                    settings["font_path"] = (
                        font_path.resolve().relative_to(self.resource_dir).as_posix()
                    )
                except ValueError:
                    settings["font_path"] = str(font_path.resolve())
        return settings

    def save_settings(self):
        if self.running or self.scan_busy or self.update_busy:
            return
        try:
            settings = self._collect_settings()
            self._save_config(settings)
        except (ConfigStoreError, OSError, ValueError) as exc:
            self._show_field_error(str(exc))
            messagebox.showerror("無法儲存設定", str(exc), parent=self.root)
            return
        self.settings = deepcopy(settings)
        self.watermark_settings = normalize_watermark_settings(self.settings)
        self.status_var.set(f"設定已儲存：{self.config_session.path.name}")
        messagebox.showinfo(
            "設定已儲存",
            f"已寫入：\n{self.config_session.path}\n\n此檔案不會加入 Git。",
            parent=self.root,
        )

    def restore_default_settings(self):
        if self.running or self.scan_busy or self.update_busy:
            return
        try:
            self.config_session.clear()
        except ConfigStoreError as exc:
            messagebox.showerror("無法還原預設", str(exc), parent=self.root)
            return

        self.settings = normalize_gui_settings(self.default_settings)
        self.targets = normalize_target_sets(self.settings)
        self.watermark_settings = normalize_watermark_settings(self.settings)
        self._apply_settings_to_widgets(self.settings)
        self.status_var.set("已還原 config.json.example 的預設設定")

    def _apply_settings_to_widgets(self, settings):
        self.output_mode_var.set(settings.get("output_mode", "source"))
        self.output_folder_var.set(settings.get("output_folder", ""))
        self.output_subfolder_var.set(settings.get("output_subfolder", ""))
        self.recursive_var.set(settings.get("include_subfolders", False))
        mode = str(settings["mode"])
        self.mode_var.set(MODE_LABELS[mode])
        for target_mode in MODE_LABELS:
            first_target = self.targets[target_mode][0]
            self.resolution_vars[target_mode].set(
                RESOLUTION_LABELS[first_target["resolution"]]
            )
            self.codec_vars[target_mode].set(CODEC_LABELS[first_target["vcodec"]])
        first_ratio = self.targets["reframe"][0]["ratio"]
        self.ratio_var.set(f"{first_ratio[0]}:{first_ratio[1]}")

        for target_mode in MODE_LABELS:
            wm = self.watermark_settings[target_mode]
            self.watermark_enabled_vars[target_mode].set(bool(wm["enabled"]))
            preferred = str(wm.get("file", ""))
            self.watermark_file_vars[target_mode].set(
                preferred
                if preferred in self.watermark_paths
                else (
                    next(iter(self.watermark_paths), "") if self.watermark_paths else ""
                )
            )
            self.watermark_position_vars[target_mode].set(
                WATERMARK_POSITION_LABELS.get(
                    wm["position"],
                    WATERMARK_POSITION_LABELS["bottom-center"],
                )
            )
            self.watermark_width_ratio_vars[target_mode].set(
                f"{float(wm['width_ratio']):g}"
            )
            self.watermark_opacity_vars[target_mode].set(
                f"{float(wm.get('opacity', DEFAULT_WATERMARK_OPACITY)):g}"
            )
            self.watermark_margin_vars[target_mode].set(str(int(wm["margin"])))

        font_path = Path(str(settings["font_path"]))
        if not font_path.is_absolute():
            font_path = self.resource_dir / font_path
        self.font_path_var.set(str(font_path))
        self.font_color_var.set(str(settings["font_color"]))
        self.top_font_size_var.set(str(settings["top_font_size"]))
        self.bottom_font_size_var.set(str(settings["bottom_font_size"]))
        self.text_margin_var.set(str(settings["text_margin"]))
        self.top_spacing_var.set(str(settings["top_spacing"]))
        self.bottom_spacing_var.set(str(settings["bottom_spacing"]))
        self.ffmpeg_var.set(tool_path("ffmpeg", settings["ffmpeg"]))
        self.ffprobe_var.set(tool_path("ffprobe", settings["ffprobe"]))
        self.workers_var.set(str(settings["workers"]))
        self.skip_existing_var.set(bool(settings["skip_existing"]))
        self.debug_var.set(bool(settings["debug"]))

        self.top_text.delete("1.0", "end")
        self.top_text.insert(
            "1.0",
            str(settings.get("top_text", "")),
        )
        self.bottom_text.delete("1.0", "end")
        self.bottom_text.insert(
            "1.0",
            str(
                settings.get(
                    "bottom_text",
                    "",
                )
            ),
        )
        for target_mode in MODE_LABELS:
            self._refresh_targets(target_mode)
        self._switch_mode()

    def save_settings_for_upgrade(self):
        settings = self._collect_settings()
        self._save_config(settings)
