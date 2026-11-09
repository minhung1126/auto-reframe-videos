# -*- coding: utf-8 -*-
"""Cross-platform Tk desktop interface for reframe and compress jobs."""

import io
import os
import queue
import sys
import threading
import time
import webbrowser
from copy import deepcopy
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, ttk
except ImportError as exc:  # pragma: no cover - depends on the Python distribution
    raise SystemExit(
        "此 Python 未包含 Tkinter。Windows 請重新安裝含 Tcl/Tk 的 Python；"
        "macOS 請安裝 python.org 版本或對應的 tkinter 套件。"
    ) from exc

from auto_reframe_core.compress import CompressConfig, VideoCompressor
from auto_reframe_core.reframe import ReframeConfig, VideoReframer
from auto_reframe_core.gui_options import (
    CODEC_KEYS_BY_LABEL,
    CODEC_LABELS,
    CODEC_OPTIONS,
    POSITION_GRID_3X3,
    RATIO_OPTIONS,
    RESOLUTION_KEYS_BY_LABEL,
    RESOLUTION_LABELS,
    RESOLUTION_OPTIONS,
    WATERMARK_POSITION_KEYS_BY_LABEL,
    WATERMARK_POSITION_LABELS,
    WATERMARK_POSITION_OPTIONS,
    list_watermark_pngs,
    parse_ratio,
)
from auto_reframe_core.watermark import (
    WATERMARK_POSITIONS,
)
from auto_reframe_core.config_store import (
    ConfigStoreError,
    ConfigSession,
    clear_config,
    save_config,
)
from auto_reframe_core.gui_settings import (
    DEFAULT_WATERMARK_MARGIN,
    DEFAULT_WATERMARK_OPACITY,
    DEFAULT_WATERMARK_POSITION,
    DEFAULT_WATERMARK_WIDTH_RATIOS,
    load_effective_settings as _load_effective_settings,
    normalize_gui_settings,
    normalize_target_sets,
    normalize_watermark_settings,
)
from auto_reframe_core.updater import (
    GITHUB_OWNER,
    GITHUB_REPOSITORY,
    UpdateError,
    can_self_update,
    check_for_update,
    launch_installer,
    prepare_update,
)
from auto_reframe_core.runtime_paths import (
    resource_root, user_data_root, is_frozen, tool_path, watermark_root, logs_root, legacy_config_paths,
)
from auto_reframe_core.job_models import JobEvent, JobRequest, BatchResult, ResultStatus
from auto_reframe_core.job_controller import JobController, JobState, EventBuffer
from auto_reframe_core.batch_runner import BatchService
from auto_reframe_core.process_runner import TaskCancelled
from auto_reframe_core.gui_preview_view import PreviewView
from auto_reframe_core.gui_files_view import FilesView
from auto_reframe_core.gui_modes_view import ModesView
from auto_reframe_core.gui_settings_view import SettingsView
from auto_reframe_core.gui_update_view import UpdateView
from auto_reframe_core.gui_activity_view import ActivityView
from auto_reframe_core.gui_job_view import JobView
from auto_reframe_core.gui_results_view import ResultsView
from auto_reframe_core.gui_dialogs import (OutputConflictDialog, build_job_confirmation_message, build_output_conflict_message, format_video_event_log)
from auto_reframe_core.gui_style import (calculate_responsive_window_geometry, copy_text_to_clipboard, choose_cjk_editor_font_family, build_cjk_editor_font, ScrollableTab)
from auto_reframe_core.gui_constants import (CJK_EDITOR_FONT_CANDIDATES, CONFIG_EXAMPLE_PATH, CONFIG_PATH, CREDIT_SYMBOL, MODE_KEYS_BY_LABEL, MODE_LABELS, OUTPUT_CONFLICT_CANCEL, OUTPUT_CONFLICT_DELETE, OUTPUT_CONFLICT_LABELS, OUTPUT_CONFLICT_OVERWRITE, OUTPUT_CONFLICT_SKIP, SCRIPT_DIR, UPDATE_ERROR_PATH, VIDEO_EXTENSIONS, WATERMARK_DIR)

from auto_reframe_core.version import __version__
from auto_reframe_core.output_plans import (
    prepare_video_batch, validate_prepared_batch, delete_planned_conflicts, output_root_for,
)
from auto_reframe_core.platform_profile import open_directory
from auto_reframe_core.video_utils import VideoProgressEvent, h264, h265


def ensure_runtime_directories(paths=None):
    """Create only per-user data directories; media stays in its original location."""
    runtime_paths = tuple(paths) if paths is not None else (WATERMARK_DIR,)
    for path in runtime_paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_effective_settings(*, store=None, migrate=True):
    return _load_effective_settings(CONFIG_EXAMPLE_PATH, store.path if store is not None else CONFIG_PATH,
                                    legacy_paths=legacy_config_paths() if migrate else (), store=store)


class AutoReframeGUI(PreviewView, ResultsView, FilesView, ModesView, SettingsView, UpdateView, ActivityView, JobView):
    def __init__(self, root: tk.Tk, *, data_root=None):
        self.root = root
        self.watermark_dir = Path(data_root) / "watermark" if data_root is not None else WATERMARK_DIR
        self.config_path = Path(data_root) / "config.json" if data_root is not None else CONFIG_PATH
        self.resource_dir = SCRIPT_DIR
        self.root.title(f"Auto Reframe Videos v{__version__}")
        window_width, window_height, min_width, min_height = (
            calculate_responsive_window_geometry(
                self.root.winfo_screenwidth(),
                self.root.winfo_screenheight(),
            )
        )
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min_width, min_height)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.editor_font = build_cjk_editor_font(self.root)

        self.event_queue = EventBuffer()
        self.job_controller = JobController(self.event_queue.put)
        self.batch_service = BatchService()
        self.last_result = None
        self.last_request = None
        self.field_error_var = tk.StringVar(value="")
        self.worker = None
        self.running = False
        self.preflight_busy = False
        self.scan_busy = False
        self.scan_worker = None
        self.update_worker = None
        self.update_busy = False
        self.installing_update = False
        self.update_info = None
        self.video_files = ()
        self.watermark_paths = {}
        self.video_progress_states = {}
        ensure_runtime_directories((self.watermark_dir,))
        self.config_session = ConfigSession(self.config_path)
        try:
            self.default_settings, self.settings = load_effective_settings(store=self.config_session, migrate=data_root is None)
        except ConfigStoreError as error:
            if not self.config_path.is_file():
                raise
            choice = messagebox.askyesnocancel("設定檔需要處理",
                f"{error}\n\n是：保留問題檔案後使用預設。否：開啟原檔查看。取消：結束程式。",
                parent=self.root)
            if choice is False:
                open_directory(self.config_path)
                raise
            if choice is not True:
                raise
            self.config_session.recover()
            self.default_settings, self.settings = load_effective_settings(store=self.config_session, migrate=False)
        self.targets = normalize_target_sets(self.settings)
        self.watermark_settings = normalize_watermark_settings(self.settings)

        self._create_variables(self.settings)
        self._build_ui()
        from auto_reframe_core.gui_style import install_shortcuts
        install_shortcuts(self)
        self._load_initial_text()
        self.refresh_watermarks()
        for mode in MODE_LABELS:
            preferred_watermark = str(self.watermark_settings[mode]["file"])
            if preferred_watermark in self.watermark_paths:
                self.watermark_file_vars[mode].set(preferred_watermark)
        self._switch_mode()
        for variable in (self.output_subfolder_var, self.ratio_var, self.font_path_var, self.font_color_var,
                         self.top_font_size_var, self.bottom_font_size_var, self.text_margin_var,
                         self.top_spacing_var, self.bottom_spacing_var,
                         *self.watermark_enabled_vars.values(), *self.watermark_file_vars.values(),
                         *self.watermark_position_vars.values(), *self.watermark_width_ratio_vars.values(),
                         *self.watermark_opacity_vars.values(), *self.watermark_margin_vars.values()):
            variable.trace_add("write", self._schedule_preview)
        self.top_text.bind("<KeyRelease>", self._schedule_preview, add="+")
        self.bottom_text.bind("<KeyRelease>", self._schedule_preview, add="+")
        self._drain_after = self.root.after(20, self._drain_events)
        self._pending_update_after = self.root.after(250, self._show_pending_update_error)
        self.root.bind("<Destroy>", self._root_destroyed, add="+")

    def _root_destroyed(self, event):
        if event.widget != self.root:
            return
        for name in ("_drain_after", "_pending_update_after", "preview_after"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    self.root.after_cancel(handle)
                except tk.TclError:
                    pass
        if getattr(self, "preview_service", None) is not None:
            self.preview_service.cancel(close=True)
        self.job_controller.cancel()


    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.style = ttk.Style()
        from auto_reframe_core.gui_style import configure_styles
        configure_styles(self.root, self.style)
        try:
            self.style.layout(
                "Compact.TRadiobutton",
                [("Radiobutton.indicator", {"sticky": ""})],
            )
            self.style.configure("Compact.TRadiobutton", padding=0)
        except tk.TclError:
            pass
        self.style.configure(
            "AnchorGrid.TFrame",
            relief="solid",
            borderwidth=1,
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        self.files_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.files_tab, text="待處理影片")
        self._build_files_tab()
        self.reframe_tab = ttk.Frame(self.notebook, padding=8)
        self.compress_tab = ttk.Frame(self.notebook, padding=8)
        self.activity_tab = ttk.Frame(self.notebook, padding=8)
        self.advanced_tab = ttk.Frame(self.notebook, padding=12)
        self.update_tab = ttk.Frame(self.notebook, padding=12)
        self.mode_tabs = {
            "reframe": self.reframe_tab,
            "compress": self.compress_tab,
        }
        self.notebook.add(self.reframe_tab, text="裁切重製")
        self.notebook.add(self.compress_tab, text="影片壓縮")
        self.notebook.add(self.activity_tab, text="處理進度")
        self.notebook.add(self.advanced_tab, text="進階設定")
        self.notebook.add(self.update_tab, text="關於／更新")

        self.target_trees = {}
        self.watermark_combos = {}
        self.watermark_position_radios = {}
        self._build_reframe_tab()
        self._build_compress_tab()
        self._build_advanced_tab()
        self._build_update_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        self._build_activity_tab()

        footer = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, wraplength=580).grid(row=0, column=0, columnspan=6, sticky="w")
        self.restore_button = ttk.Button(
            footer, text="還原預設", command=self.restore_default_settings
        )
        self.restore_button.grid(row=1, column=1, padx=(8, 0))
        self.save_button = ttk.Button(
            footer, text="儲存設定", command=self.save_settings
        )
        self.save_button.grid(row=1, column=2, padx=(8, 0))
        self.start_button = ttk.Button(footer, text="開始處理", command=self.start_job)
        self.start_button.grid(row=1, column=3, padx=(8, 0))
        self.stop_button = ttk.Button(footer, text="停止處理", command=self.stop_job, state="disabled")
        self.stop_button.grid(row=1, column=4, padx=(8, 0))
        self.preview_button = ttk.Button(footer, text="預覽", command=self.open_preview)
        self.preview_button.grid(row=1, column=5, padx=(8, 0))
        ttk.Label(footer, textvariable=self.field_error_var, foreground="#b00020", wraplength=580).grid(
            row=2, column=0, columnspan=6, sticky="w")


    def open_config(self):
        try:
            path = getattr(self, "config_path", CONFIG_PATH)
            if not path.exists():
                self._save_config(self.default_settings)
            open_directory(path)
        except (ConfigStoreError, OSError, ValueError) as exc:
            messagebox.showerror("無法開啟設定檔", str(exc), parent=self.root)


    def _save_config(self, settings):
        if hasattr(self, "config_session"):
            self.config_session.save(settings)
        else:
            save_config(CONFIG_PATH, settings)


def main():
    from auto_reframe_core.gui_style import create_gui_root
    root = create_gui_root()
    try:
        app = AutoReframeGUI(root)
    except (ConfigStoreError, OSError, ValueError) as exc:
        messagebox.showerror("無法啟動", str(exc), parent=root)
        root.destroy()
        return
    try:
        root.mainloop()
    finally:
        # Also reap children after an unexpected Tk exit or console interrupt.
        if app.job_controller.context is not None:
            app.job_controller.context.cancellation.cancel()
        if getattr(app, "preview_service", None) is not None:
            app.preview_service.cancel(close=True)
            if app.preview_service.context is not None:
                app.preview_service.context.cancellation.cancel()


if __name__ == "__main__":
    main()
