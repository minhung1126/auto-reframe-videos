# -*- coding: utf-8 -*-
"""Cross-platform Tk desktop interface for reframe and compress jobs."""

import io
import os
import queue
import sys
import threading
import webbrowser
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
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
    RATIO_OPTIONS,
    RESOLUTION_KEYS_BY_LABEL,
    RESOLUTION_LABELS,
    RESOLUTION_OPTIONS,
    list_watermark_pngs,
    parse_ratio,
)
from auto_reframe_core.config_store import (
    ConfigStoreError,
    clear_config,
    load_config,
    save_config,
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
from auto_reframe_core.version import __version__
from auto_reframe_core.video_utils import h264, h265


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = SCRIPT_DIR / "config.json"
CONFIG_EXAMPLE_PATH = SCRIPT_DIR / "config.json.example"
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
WATERMARK_DIR = SCRIPT_DIR / "watermark"
UPDATE_ERROR_PATH = SCRIPT_DIR / "update-error.log"
CREDIT_SYMBOL = "©"
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"
}

MODE_LABELS = {
    "reframe": "比例裁切／直式重製",
    "compress": "影片壓縮／縮小解析度",
}
MODE_KEYS_BY_LABEL = {label: key for key, label in MODE_LABELS.items()}


def ensure_runtime_directories(paths=None):
    """Create the fixed runtime directories before any job starts."""
    runtime_paths = tuple(paths) if paths is not None else (
        INPUT_DIR,
        OUTPUT_DIR,
        WATERMARK_DIR,
    )
    for path in runtime_paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_effective_settings():
    """Load committed defaults, then overlay the optional local config."""
    defaults = load_config(CONFIG_EXAMPLE_PATH)
    if defaults is None:
        raise ConfigStoreError(f"找不到預設設定: {CONFIG_EXAMPLE_PATH}")
    effective = deepcopy(defaults)
    saved = load_config(CONFIG_PATH)
    if saved:
        effective.update(saved)
    return defaults, effective


def normalize_target_sets(settings: dict) -> dict:
    """Validate target data loaded from JSON before it reaches the GUI."""
    raw_targets = settings.get("targets")
    if not isinstance(raw_targets, dict):
        raise ConfigStoreError("設定檔的 targets 必須是物件。")

    final_ratio_value = settings.get("final_ratio")
    if not isinstance(final_ratio_value, (list, tuple)) or len(final_ratio_value) != 2:
        raise ConfigStoreError("設定檔的 final_ratio 必須是 [寬, 高]。")
    final_ratio = (int(final_ratio_value[0]), int(final_ratio_value[1]))
    if final_ratio[0] <= 0 or final_ratio[1] <= 0:
        raise ConfigStoreError("設定檔的 final_ratio 必須大於 0。")

    normalized = {"reframe": [], "compress": []}
    for mode in normalized:
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
                ratio_value = entry.get("ratio")
                if not isinstance(ratio_value, (list, tuple)) or len(ratio_value) != 2:
                    raise ConfigStoreError("重製 target 的 ratio 必須是 [寬, 高]。")
                ratio = (int(ratio_value[0]), int(ratio_value[1]))
                if ratio[0] <= 0 or ratio[1] <= 0:
                    raise ConfigStoreError("重製 target 的 ratio 必須大於 0。")
                if ratio[0] / ratio[1] < final_ratio[0] / final_ratio[1]:
                    raise ConfigStoreError("中央影片比例不可比 final_ratio 更窄高。")
                target["ratio"] = ratio
            if target not in normalized[mode]:
                normalized[mode].append(target)
    return normalized


class QueueStream(io.TextIOBase):
    """Send worker-thread text output to the Tk event queue."""

    def __init__(self, event_queue: queue.Queue):
        super().__init__()
        self.event_queue = event_queue

    def write(self, value):
        if value:
            self.event_queue.put(("log", str(value)))
        return len(value)

    def flush(self):
        return None


def _read_optional_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8-sig").replace("\r", "").rstrip("\n")
    except (OSError, UnicodeError):
        return ""


def copy_text_to_clipboard(root, text: str) -> None:
    """Copy text through Tk so the shortcut works on Windows and macOS."""
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update_idletasks()


class AutoReframeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Auto Reframe Videos v{__version__}")
        self.root.geometry("1040x850")
        self.root.minsize(900, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.event_queue = queue.Queue()
        self.worker = None
        self.running = False
        self.update_worker = None
        self.update_busy = False
        self.installing_update = False
        self.update_info = None
        self.watermark_paths = {}
        ensure_runtime_directories()
        self.default_settings, self.settings = load_effective_settings()
        self.targets = normalize_target_sets(self.settings)

        self._create_variables(self.settings)
        self._build_ui()
        self._load_initial_text()
        self.refresh_watermarks()
        preferred_watermark = str(self.settings.get("watermark_file", ""))
        if preferred_watermark in self.watermark_paths:
            self.watermark_var.set(preferred_watermark)
        self._switch_mode()
        self.root.after(80, self._drain_events)
        self.root.after(250, self._show_pending_update_error)

    def _create_variables(self, settings):
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

        self.watermark_enabled_var = tk.BooleanVar(
            value=bool(settings.get("watermark_enabled"))
        )
        self.watermark_var = tk.StringVar(value="")

        font_path = Path(str(settings["font_path"]))
        if not font_path.is_absolute():
            font_path = SCRIPT_DIR / font_path
        self.font_path_var = tk.StringVar(value=str(font_path))
        self.font_color_var = tk.StringVar(value=str(settings["font_color"]))
        self.top_font_size_var = tk.StringVar(value=str(settings["top_font_size"]))
        self.bottom_font_size_var = tk.StringVar(value=str(settings["bottom_font_size"]))
        self.text_margin_var = tk.StringVar(value=str(settings["text_margin"]))
        self.top_spacing_var = tk.StringVar(value=str(settings["top_spacing"]))
        self.bottom_spacing_var = tk.StringVar(value=str(settings["bottom_spacing"]))

        self.ffmpeg_var = tk.StringVar(value=str(settings["ffmpeg"]))
        self.ffprobe_var = tk.StringVar(value=str(settings["ffprobe"]))
        self.workers_var = tk.StringVar(value=str(settings["workers"]))
        self.skip_existing_var = tk.BooleanVar(value=bool(settings["skip_existing"]))
        self.debug_var = tk.BooleanVar(value=bool(settings["debug"]))
        self.status_var = tk.StringVar(value="準備就緒")
        self.update_status_var = tk.StringVar(value="尚未檢查更新")

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        self.reframe_tab = ttk.Frame(self.notebook, padding=8)
        self.compress_tab = ttk.Frame(self.notebook, padding=8)
        self.advanced_tab = ttk.Frame(self.notebook, padding=12)
        self.update_tab = ttk.Frame(self.notebook, padding=12)
        self.mode_tabs = {
            "reframe": self.reframe_tab,
            "compress": self.compress_tab,
        }
        self.notebook.add(self.reframe_tab, text="裁切重製")
        self.notebook.add(self.compress_tab, text="影片壓縮")
        self.notebook.add(self.advanced_tab, text="進階設定")
        self.notebook.add(self.update_tab, text="關於／更新")

        self.target_trees = {}
        self.watermark_combos = {}
        self._build_reframe_tab()
        self._build_compress_tab()
        self._build_advanced_tab()
        self._build_update_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        log_frame = ttk.LabelFrame(self.root, text="執行日誌", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        self.root.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap="word", state="disabled", font=("TkFixedFont", 9)
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.restore_button = ttk.Button(
            footer, text="還原預設", command=self.restore_default_settings
        )
        self.restore_button.grid(row=0, column=1, padx=(8, 0))
        self.save_button = ttk.Button(
            footer, text="儲存設定", command=self.save_settings
        )
        self.save_button.grid(row=0, column=2, padx=(8, 0))
        self.start_button = ttk.Button(footer, text="開始處理", command=self.start_job)
        self.start_button.grid(row=0, column=3, padx=(8, 0))

    def _build_reframe_tab(self):
        self.reframe_tab.columnconfigure(0, weight=1)
        self.reframe_tab.rowconfigure(0, weight=1)
        self.reframe_notebook = ttk.Notebook(self.reframe_tab)
        self.reframe_notebook.grid(row=0, column=0, sticky="nsew")

        self.reframe_output_tab = ttk.Frame(self.reframe_notebook, padding=10)
        self.context_tab = ttk.Frame(self.reframe_notebook, padding=10)
        self.text_style_tab = ttk.Frame(self.reframe_notebook, padding=10)
        self.reframe_notebook.add(self.reframe_output_tab, text="輸出設定")
        self.reframe_notebook.add(self.context_tab, text="編輯上下文")
        self.reframe_notebook.add(self.text_style_tab, text="文字樣式")

        self._build_mode_output_tab(self.reframe_output_tab, "reframe")
        self._build_context_tab()
        self._build_text_style_tab()

    def _build_compress_tab(self):
        self.compress_tab.columnconfigure(0, weight=1)
        self.compress_tab.rowconfigure(0, weight=1)
        self.compress_notebook = ttk.Notebook(self.compress_tab)
        self.compress_notebook.grid(row=0, column=0, sticky="nsew")

        self.compress_output_tab = ttk.Frame(self.compress_notebook, padding=10)
        self.compress_notebook.add(self.compress_output_tab, text="輸出設定")
        self._build_mode_output_tab(self.compress_output_tab, "compress")

    def _build_mode_output_tab(self, parent, mode):
        parent.columnconfigure(0, weight=1)

        job = ttk.LabelFrame(parent, text="固定工作資料夾", padding=10)
        job.grid(row=0, column=0, sticky="ew")
        job.columnconfigure(1, weight=1)

        ttk.Label(job, text="輸入資料夾").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(
            job,
            text=str(INPUT_DIR),
            relief="sunken",
            anchor="w",
            padding=(5, 3),
        ).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(job, text="輸出資料夾").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(
            job,
            text=str(OUTPUT_DIR),
            relief="sunken",
            anchor="w",
            padding=(5, 3),
        ).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=4
        )

        targets = ttk.LabelFrame(parent, text="輸出目標（可加入多個組合）", padding=10)
        targets.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        targets.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        target_tree = ttk.Treeview(
            targets,
            columns=("ratio", "resolution", "codec"),
            show="headings",
            height=6,
            selectmode="extended",
        )
        target_tree.heading(
            "ratio",
            text="中央影片裁切比例" if mode == "reframe" else "輸出比例",
        )
        target_tree.heading("resolution", text="解析度上限（不放大）")
        target_tree.heading("codec", text="視訊編碼")
        target_tree.column("ratio", width=160, anchor="center")
        target_tree.column("resolution", width=260, anchor="center")
        target_tree.column("codec", width=200, anchor="center")
        target_tree.grid(row=0, column=0, columnspan=6, sticky="nsew")
        self.target_trees[mode] = target_tree

        ttk.Label(targets, text="比例").grid(row=1, column=0, sticky="w", pady=(10, 0))
        if mode == "reframe":
            ttk.Combobox(
                targets,
                textvariable=self.ratio_var,
                values=RATIO_OPTIONS,
                width=12,
            ).grid(row=2, column=0, sticky="ew", padx=(0, 8))
        else:
            ttk.Label(
                targets,
                text="保留來源比例",
                relief="sunken",
                anchor="center",
                padding=(5, 3),
            ).grid(row=2, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(targets, text="解析度").grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Combobox(
            targets,
            textvariable=self.resolution_vars[mode],
            values=tuple(label for _, label in RESOLUTION_OPTIONS),
            state="readonly",
            width=28,
        ).grid(row=2, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(targets, text="Codec").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Combobox(
            targets,
            textvariable=self.codec_vars[mode],
            values=tuple(label for _, label in CODEC_OPTIONS),
            state="readonly",
            width=20,
        ).grid(row=2, column=2, sticky="ew", padx=(0, 8))

        ttk.Button(
            targets,
            text="加入目標",
            command=lambda selected_mode=mode: self._add_target(selected_mode),
        ).grid(
            row=2, column=3, padx=(0, 8)
        )
        ttk.Button(
            targets,
            text="移除選取",
            command=lambda selected_mode=mode: self._remove_targets(selected_mode),
        ).grid(
            row=2, column=4, padx=(0, 8)
        )
        ttk.Button(
            targets,
            text="恢復預設",
            command=lambda selected_mode=mode: self._reset_targets(selected_mode),
        ).grid(
            row=2, column=5
        )

        watermark = ttk.LabelFrame(parent, text="PNG 浮水印", padding=10)
        watermark.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        watermark.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            watermark,
            text="蓋浮水印",
            variable=self.watermark_enabled_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        watermark_combo = ttk.Combobox(
            watermark,
            textvariable=self.watermark_var,
            state="readonly",
        )
        self.watermark_combos[mode] = watermark_combo
        watermark_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(watermark, text="重新整理", command=self.refresh_watermarks).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Label(
            watermark,
            text="來源：watermark/*.png；預設下方中央，寬度為輸出畫面的 32%，底部距離以 FHD 56px 等比縮放。",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_context_tab(self):
        self.context_tab.columnconfigure(0, weight=1)
        ttk.Label(
            self.context_tab,
            text="最終畫布固定為 9:16，並永遠維持「上方文字／中央影片／下方文字」三層。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        credit = ttk.LabelFrame(
            self.context_tab,
            text="Credit／版權符號",
            padding=(10, 6),
        )
        credit.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(credit, text=CREDIT_SYMBOL, font=("TkDefaultFont", 16)).grid(
            row=0, column=0, padx=(0, 12)
        )
        ttk.Button(
            credit,
            text="複製 ©",
            command=self._copy_credit_symbol,
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(
            credit,
            text="插入上方文字",
            command=lambda: self._insert_credit_symbol(self.top_text),
        ).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(
            credit,
            text="插入下方文字",
            command=lambda: self._insert_credit_symbol(self.bottom_text),
        ).grid(row=0, column=3)

        ttk.Label(self.context_tab, text="上方文字（可多行）").grid(
            row=2, column=0, sticky="w"
        )
        self.top_text = scrolledtext.ScrolledText(self.context_tab, height=5, wrap="word")
        self.top_text.grid(row=3, column=0, sticky="nsew", pady=(4, 10))

        ttk.Label(self.context_tab, text="下方文字（可多行）").grid(
            row=4, column=0, sticky="w"
        )
        self.bottom_text = scrolledtext.ScrolledText(self.context_tab, height=4, wrap="word")
        self.bottom_text.grid(row=5, column=0, sticky="nsew", pady=(4, 0))
        self.context_tab.rowconfigure(3, weight=1)
        self.context_tab.rowconfigure(5, weight=1)

    def _build_text_style_tab(self):
        self.text_style_tab.columnconfigure(0, weight=1)
        ttk.Label(
            self.text_style_tab,
            text="此頁設定只套用到「裁切重製」的上、下方文字。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        style = ttk.LabelFrame(self.text_style_tab, text="文字樣式", padding=10)
        style.grid(row=1, column=0, sticky="ew")
        style.columnconfigure(1, weight=1)

        ttk.Label(style, text="字型").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(style, textvariable=self.font_path_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(style, text="瀏覽…", command=self._browse_font).grid(
            row=0, column=2, padx=(8, 0)
        )
        self._labeled_entry(style, "顏色", self.font_color_var, 1, 0)
        self._labeled_entry(style, "上方字級", self.top_font_size_var, 1, 2)
        self._labeled_entry(style, "下方字級", self.bottom_font_size_var, 1, 4)
        self._labeled_entry(style, "影片邊距", self.text_margin_var, 2, 0)
        self._labeled_entry(style, "上方行距", self.top_spacing_var, 2, 2)
        self._labeled_entry(style, "下方行距", self.bottom_spacing_var, 2, 4)

    @staticmethod
    def _labeled_entry(parent, label, variable, row, column):
        ttk.Label(parent, text=label).grid(
            row=row, column=column, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 12), pady=3
        )

    def _build_advanced_tab(self):
        self.advanced_tab.columnconfigure(1, weight=1)
        ttk.Label(
            self.advanced_tab,
            text="硬體編碼會自動探測：Windows 優先 NVENC／AMF／QSV，macOS 優先 VideoToolbox，否則回退軟體。",
            wraplength=850,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        self._advanced_path_row("FFmpeg", self.ffmpeg_var, 1)
        self._advanced_path_row("FFprobe", self.ffprobe_var, 2)
        ttk.Label(self.advanced_tab, text="平行工作數").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Entry(self.advanced_tab, textvariable=self.workers_var).grid(
            row=3, column=1, sticky="w", pady=5
        )
        ttk.Label(self.advanced_tab, text="0 = 自動判斷").grid(
            row=3, column=2, sticky="w", padx=(8, 0)
        )
        ttk.Checkbutton(
            self.advanced_tab,
            text="跳過已存在的輸出",
            variable=self.skip_existing_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(
            self.advanced_tab,
            text="輸出 FFmpeg 除錯日誌",
            variable=self.debug_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=5)

    def _build_update_tab(self):
        self.update_tab.columnconfigure(0, weight=1)
        about = ttk.LabelFrame(self.update_tab, text="Auto Reframe Videos", padding=12)
        about.grid(row=0, column=0, sticky="ew")
        about.columnconfigure(1, weight=1)
        ttk.Label(about, text="目前版本").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(about, text=f"v{__version__}").grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(about, text="更新來源").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(
            about,
            text=f"github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases",
        ).grid(row=1, column=1, sticky="w", pady=4)

        updater = ttk.LabelFrame(self.update_tab, text="軟體更新", padding=12)
        updater.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        updater.columnconfigure(0, weight=1)
        self.update_tab.rowconfigure(1, weight=1)
        ttk.Label(
            updater,
            textvariable=self.update_status_var,
            wraplength=850,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        self.check_update_button = ttk.Button(
            updater,
            text="檢查更新",
            command=self.check_for_updates,
        )
        self.check_update_button.grid(row=1, column=0, sticky="w", pady=(10, 8))
        self.install_update_button = ttk.Button(
            updater,
            text="下載並安裝",
            command=self.install_update,
            state="disabled",
        )
        self.install_update_button.grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 8)
        )
        self.open_release_button = ttk.Button(
            updater,
            text="開啟 Release 頁面",
            command=self.open_release_page,
        )
        self.open_release_button.grid(
            row=1, column=2, sticky="w", padx=(8, 0), pady=(10, 8)
        )

        allowed, reason = can_self_update(SCRIPT_DIR)
        environment_text = (
            "此安裝可自動更新；安裝時會備份舊版，並保留 config.json、"
            "input/、output/、watermark/ 與上下方文字。"
            if allowed
            else reason
        )
        ttk.Label(
            updater,
            text=environment_text,
            foreground="#555555",
            wraplength=850,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(updater, text="版本說明").grid(
            row=3, column=0, columnspan=3, sticky="w"
        )
        self.release_notes = scrolledtext.ScrolledText(
            updater,
            height=12,
            wrap="word",
            state="disabled",
            font=("TkDefaultFont", 9),
        )
        self.release_notes.grid(
            row=4, column=0, columnspan=3, sticky="nsew", pady=(4, 0)
        )
        updater.rowconfigure(4, weight=1)

    def _advanced_path_row(self, label, variable, row):
        ttk.Label(self.advanced_tab, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Entry(self.advanced_tab, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            self.advanced_tab,
            text="選擇…",
            command=lambda target=variable: self._browse_executable(target),
        ).grid(row=row, column=2, padx=(8, 0), pady=5)

    def _load_initial_text(self):
        top_text = self.settings.get(
            "top_text", _read_optional_text(SCRIPT_DIR / "top_text.txt")
        )
        bottom_text = self.settings.get(
            "bottom_text", _read_optional_text(SCRIPT_DIR / "bottom_text.txt")
        )
        self.top_text.insert("1.0", str(top_text))
        self.bottom_text.insert("1.0", str(bottom_text))

    def _browse_font(self):
        selected = filedialog.askopenfilename(
            title="選擇字型",
            initialdir=str(Path(self.font_path_var.get()).parent),
            filetypes=(("TrueType / OpenType", "*.ttf *.otf"), ("所有檔案", "*.*")),
        )
        if selected:
            self.font_path_var.set(selected)

    def _browse_executable(self, target_var):
        selected = filedialog.askopenfilename(title="選擇執行檔")
        if selected:
            target_var.set(selected)

    def _set_release_notes(self, value):
        self.release_notes.configure(state="normal")
        self.release_notes.delete("1.0", "end")
        self.release_notes.insert("1.0", value)
        self.release_notes.configure(state="disabled")

    def _show_pending_update_error(self):
        if not UPDATE_ERROR_PATH.is_file():
            return
        try:
            error = UPDATE_ERROR_PATH.read_text(encoding="utf-8")[:16000].strip()
            UPDATE_ERROR_PATH.unlink()
        except (OSError, UnicodeError) as exc:
            error = f"無法讀取更新錯誤紀錄: {exc}"
        if error:
            self.status_var.set("上一個軟體更新未完成")
            messagebox.showerror(
                "軟體更新未完成",
                error,
                parent=self.root,
            )

    def open_release_page(self):
        url = (
            self.update_info.release_url
            if self.update_info is not None
            else f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases"
        )
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error as exc:
            messagebox.showerror("無法開啟瀏覽器", str(exc), parent=self.root)
            return
        if not opened:
            messagebox.showinfo(
                "Release 頁面",
                url,
                parent=self.root,
            )

    def check_for_updates(self):
        if self.update_busy:
            return
        self.update_busy = True
        self.update_info = None
        self.check_update_button.configure(state="disabled")
        self.install_update_button.configure(state="disabled")
        self.update_status_var.set("正在連線 GitHub 檢查最新正式版本…")
        self.status_var.set("正在檢查軟體更新…")
        self.update_worker = threading.Thread(
            target=self._run_update_check,
            name="update-check",
            daemon=True,
        )
        self.update_worker.start()

    def _run_update_check(self):
        try:
            info = check_for_update()
        except BaseException as exc:
            self.event_queue.put(("update_check_error", f"{type(exc).__name__}: {exc}"))
        else:
            self.event_queue.put(("update_check_done", info))

    def _finish_update_check(self, info):
        self.update_busy = False
        self.update_info = info
        self.check_update_button.configure(state="normal")
        immutable_text = "不可變 Release" if info.immutable else "一般 Release"
        if info.available:
            self.update_status_var.set(
                f"有新版本 v{info.latest_version}（目前 v{info.current_version}，"
                f"{immutable_text}）。"
            )
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if allowed and not self.running:
                self.install_update_button.configure(state="normal")
            self.status_var.set(f"可更新至 v{info.latest_version}")
        else:
            self.update_status_var.set(
                f"目前已是最新版 v{info.current_version}（GitHub: "
                f"v{info.latest_version}）。"
            )
            self.install_update_button.configure(state="disabled")
            self.status_var.set("目前已是最新版")
        notes = info.notes.strip() or "此版本沒有附加版本說明。"
        self._set_release_notes(notes)

    def _finish_update_check_error(self, error):
        self.update_busy = False
        self.check_update_button.configure(state="normal")
        self.install_update_button.configure(state="disabled")
        self.update_status_var.set("更新檢查失敗。")
        self.status_var.set("無法檢查更新")
        self._set_release_notes(error)
        messagebox.showerror("無法檢查更新", error, parent=self.root)

    def install_update(self):
        if self.update_busy or self.update_info is None or not self.update_info.available:
            return
        if self.running:
            messagebox.showwarning(
                "影片處理中",
                "請等待影片處理完成後再安裝更新。",
                parent=self.root,
            )
            return
        allowed, reason = can_self_update(SCRIPT_DIR)
        if not allowed:
            messagebox.showinfo("此環境不可自動更新", reason, parent=self.root)
            return
        confirmed = messagebox.askyesno(
            "安裝軟體更新",
            f"將下載並安裝 v{self.update_info.latest_version}。\n\n"
            "更新檔會先驗證 SHA-256，舊程式會備份；個人設定、影片、"
            "浮水印與上下方文字不會被覆寫。完成後程式會自動重新啟動。\n\n"
            "要繼續嗎？",
            parent=self.root,
        )
        if not confirmed:
            return
        try:
            settings = self._collect_settings()
            normalize_target_sets(settings)
            save_config(CONFIG_PATH, settings)
            self.settings = deepcopy(settings)
        except (ConfigStoreError, OSError, ValueError) as exc:
            messagebox.showerror(
                "無法在更新前儲存設定",
                str(exc),
                parent=self.root,
            )
            return

        self.update_busy = True
        self.check_update_button.configure(state="disabled")
        self.install_update_button.configure(state="disabled")
        self._set_footer_buttons("disabled")
        self.update_status_var.set("正在下載並驗證更新檔…")
        self.status_var.set("正在準備軟體更新…")
        self.update_worker = threading.Thread(
            target=self._run_update_prepare,
            args=(self.update_info,),
            name="update-download",
            daemon=True,
        )
        self.update_worker.start()

    def _run_update_prepare(self, info):
        def report_progress(received, total):
            self.event_queue.put(("update_progress", (received, total)))

        try:
            staged = prepare_update(info, SCRIPT_DIR, progress=report_progress)
        except BaseException as exc:
            self.event_queue.put(("update_install_error", f"{type(exc).__name__}: {exc}"))
        else:
            self.event_queue.put(("update_staged", staged))

    def _finish_update_staged(self, staged):
        messagebox.showinfo(
            "更新已準備完成",
            "程式將關閉、安裝已驗證的更新，然後自動重新啟動。",
            parent=self.root,
        )
        restart_command = (
            [sys.executable]
            if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "auto_reframe_core", "gui"]
        )
        try:
            launch_installer(
                staged,
                SCRIPT_DIR,
                restart_command,
                parent_pid=os.getpid(),
            )
        except UpdateError as exc:
            self._finish_update_install_error(str(exc))
            return
        self.installing_update = True
        self.update_status_var.set("更新安裝程式已啟動，正在關閉…")
        self.status_var.set("即將安裝更新並重新啟動")
        self.root.after(200, self.root.destroy)

    def _finish_update_install_error(self, error):
        self.update_busy = False
        self.check_update_button.configure(state="normal")
        if self.update_info is not None and self.update_info.available:
            self.install_update_button.configure(state="normal")
        self._set_footer_buttons("normal")
        self.update_status_var.set("更新安裝失敗。")
        self.status_var.set("更新安裝失敗")
        messagebox.showerror("無法安裝更新", error, parent=self.root)

    def _copy_credit_symbol(self):
        try:
            copy_text_to_clipboard(self.root, CREDIT_SYMBOL)
        except tk.TclError as exc:
            messagebox.showerror("無法複製", str(exc), parent=self.root)
            return
        self.status_var.set("已複製 © 到剪貼簿")

    def _insert_credit_symbol(self, target):
        target.insert("insert", CREDIT_SYMBOL)
        target.focus_set()
        self.status_var.set("已插入 ©")

    def _mode_key(self):
        return MODE_KEYS_BY_LABEL[self.mode_var.get()]

    def _on_main_tab_changed(self, _event=None):
        selected_tab = self.notebook.select()
        for mode, tab in self.mode_tabs.items():
            if selected_tab == str(tab):
                self.mode_var.set(MODE_LABELS[mode])
                self._refresh_targets(mode)
                return

    def _switch_mode(self):
        mode = self._mode_key()
        self.notebook.select(self.mode_tabs[mode])
        self._refresh_targets(mode)

    def _refresh_targets(self, mode=None):
        mode = mode or self._mode_key()
        target_tree = self.target_trees[mode]
        for item in target_tree.get_children():
            target_tree.delete(item)
        for target in self.targets[mode]:
            ratio = (
                f"{target['ratio'][0]}:{target['ratio'][1]}"
                if mode == "reframe"
                else "保留來源比例"
            )
            target_tree.insert(
                "",
                "end",
                values=(
                    ratio,
                    RESOLUTION_LABELS[target["resolution"]],
                    CODEC_LABELS[target["vcodec"]],
                ),
            )

    def _add_target(self, mode):
        resolution = RESOLUTION_KEYS_BY_LABEL[self.resolution_vars[mode].get()]
        codec = CODEC_KEYS_BY_LABEL[self.codec_vars[mode].get()]
        if mode == "reframe":
            try:
                ratio = parse_ratio(self.ratio_var.get())
                if ratio[0] / ratio[1] < 9 / 16:
                    raise ValueError("中央影片比例不可比最終 9:16 畫布更窄高。")
            except ValueError as exc:
                messagebox.showerror("比例無效", str(exc), parent=self.root)
                return
            target = {"ratio": ratio, "resolution": resolution, "vcodec": codec}
        else:
            target = {"resolution": resolution, "vcodec": codec}

        if target in self.targets[mode]:
            messagebox.showinfo("目標已存在", "相同的輸出目標已在清單中。", parent=self.root)
            return
        self.targets[mode].append(target)
        self._refresh_targets(mode)

    def _remove_targets(self, mode):
        target_tree = self.target_trees[mode]
        indices = sorted(
            (target_tree.index(item) for item in target_tree.selection()),
            reverse=True,
        )
        for index in indices:
            self.targets[mode].pop(index)
        self._refresh_targets(mode)

    def _reset_targets(self, mode):
        default_targets = normalize_target_sets(self.default_settings)
        self.targets[mode] = default_targets[mode]
        first_target = self.targets[mode][0]
        self.resolution_vars[mode].set(RESOLUTION_LABELS[first_target["resolution"]])
        self.codec_vars[mode].set(CODEC_LABELS[first_target["vcodec"]])
        if mode == "reframe":
            ratio = first_target["ratio"]
            self.ratio_var.set(f"{ratio[0]}:{ratio[1]}")
        self._refresh_targets(mode)

    def refresh_watermarks(self):
        try:
            WATERMARK_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("無法建立 watermark 資料夾", str(exc), parent=self.root)
            return

        files = list_watermark_pngs(WATERMARK_DIR)
        self.watermark_paths = {item.name: item for item in files}
        names = tuple(self.watermark_paths)
        current = self.watermark_var.get()
        for watermark_combo in self.watermark_combos.values():
            watermark_combo.configure(
                values=names,
                state="readonly" if names else "disabled",
            )
        if current not in self.watermark_paths:
            self.watermark_var.set(names[0] if names else "")
        if not names:
            self.status_var.set("watermark/ 尚無 PNG；需要浮水印時請先放入圖片")

    def _build_config(self):
        mode = self._mode_key()
        ensure_runtime_directories()
        input_dir = INPUT_DIR.resolve()
        output_dir = OUTPUT_DIR.resolve()
        videos = [
            item for item in input_dir.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not videos:
            raise ValueError("輸入資料夾中沒有支援的影片檔。")
        if not self.targets[mode]:
            raise ValueError("請至少加入一個輸出目標。")

        output_dir.mkdir(parents=True, exist_ok=True)
        workers = int(self.workers_var.get().strip())
        if workers < 0:
            raise ValueError("平行工作數不可小於 0。")

        watermark_enabled = self.watermark_enabled_var.get()
        watermark_path = self.watermark_paths.get(self.watermark_var.get())
        if watermark_enabled and watermark_path is None:
            raise ValueError("已啟用浮水印，但 watermark/ 中沒有可用的 PNG。")

        common = dict(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            targets=[dict(target) for target in self.targets[mode]],
            ffmpeg_path=self.ffmpeg_var.get().strip() or "ffmpeg",
            ffprobe_path=self.ffprobe_var.get().strip() or "ffprobe",
            skip_existing=self.skip_existing_var.get(),
            max_workers=workers,
            debug=self.debug_var.get(),
            watermark_enabled=watermark_enabled,
            watermark_file=str(watermark_path) if watermark_path else "",
            watermark_position=str(self.settings["watermark_position"]),
            watermark_width_ratio=float(self.settings["watermark_width_ratio"]),
            watermark_opacity=float(self.settings["watermark_opacity"]),
            watermark_margin=int(self.settings["watermark_margin"]),
        )
        if mode == "compress":
            return mode, CompressConfig(**common)

        font_path = Path(self.font_path_var.get()).expanduser().resolve()
        if not font_path.is_file():
            raise ValueError(f"找不到字型檔: {font_path}")
        config = ReframeConfig(
            **common,
            final_ratio=tuple(int(value) for value in self.settings["final_ratio"]),
            top_text_override=self.top_text.get("1.0", "end-1c"),
            bottom_text_override=self.bottom_text.get("1.0", "end-1c"),
            font_path=str(font_path),
            font_color=self.font_color_var.get().strip() or "white",
            top_font_size=int(self.top_font_size_var.get()),
            bottom_font_size=int(self.bottom_font_size_var.get()),
            text_margin=int(self.text_margin_var.get()),
            top_text_line_spacing_ratio=float(self.top_spacing_var.get()),
            bottom_text_line_spacing_ratio=float(self.bottom_spacing_var.get()),
        )
        return mode, config

    def _collect_settings(self) -> dict:
        """Collect GUI state without mutable input/output directory settings."""
        font_path = Path(self.font_path_var.get()).expanduser()
        try:
            font_value = font_path.resolve().relative_to(SCRIPT_DIR).as_posix()
        except ValueError:
            font_value = str(font_path.resolve())

        return {
            "mode": self._mode_key(),
            "targets": deepcopy(self.targets),
            "final_ratio": list(self.settings["final_ratio"]),
            "watermark_enabled": bool(self.watermark_enabled_var.get()),
            "watermark_file": self.watermark_var.get(),
            "watermark_position": str(self.settings["watermark_position"]),
            "watermark_width_ratio": float(self.settings["watermark_width_ratio"]),
            "watermark_opacity": float(self.settings["watermark_opacity"]),
            "watermark_margin": int(self.settings["watermark_margin"]),
            "top_text": self.top_text.get("1.0", "end-1c"),
            "bottom_text": self.bottom_text.get("1.0", "end-1c"),
            "font_path": font_value,
            "font_color": self.font_color_var.get().strip() or "white",
            "top_font_size": int(self.top_font_size_var.get()),
            "bottom_font_size": int(self.bottom_font_size_var.get()),
            "text_margin": int(self.text_margin_var.get()),
            "top_spacing": float(self.top_spacing_var.get()),
            "bottom_spacing": float(self.bottom_spacing_var.get()),
            "ffmpeg": self.ffmpeg_var.get().strip() or "ffmpeg",
            "ffprobe": self.ffprobe_var.get().strip() or "ffprobe",
            "workers": int(self.workers_var.get()),
            "skip_existing": bool(self.skip_existing_var.get()),
            "debug": bool(self.debug_var.get()),
        }

    def save_settings(self):
        try:
            settings = self._collect_settings()
            normalize_target_sets(settings)
            save_config(CONFIG_PATH, settings)
        except (ConfigStoreError, OSError, ValueError) as exc:
            messagebox.showerror("無法儲存設定", str(exc), parent=self.root)
            return
        self.settings = deepcopy(settings)
        self.status_var.set(f"設定已儲存：{CONFIG_PATH.name}")
        messagebox.showinfo(
            "設定已儲存",
            f"已寫入：\n{CONFIG_PATH}\n\n此檔案不會加入 Git。",
            parent=self.root,
        )

    def restore_default_settings(self):
        try:
            clear_config(CONFIG_PATH)
        except ConfigStoreError as exc:
            messagebox.showerror("無法還原預設", str(exc), parent=self.root)
            return

        self.settings = deepcopy(self.default_settings)
        self.targets = normalize_target_sets(self.settings)
        self._apply_settings_to_widgets(self.settings)
        self.status_var.set("已還原 config.json.example 的預設設定")

    def _apply_settings_to_widgets(self, settings):
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
        self.watermark_enabled_var.set(bool(settings["watermark_enabled"]))
        preferred = str(settings.get("watermark_file", ""))
        self.watermark_var.set(
            preferred if preferred in self.watermark_paths
            else next(iter(self.watermark_paths), "")
        )

        font_path = Path(str(settings["font_path"]))
        if not font_path.is_absolute():
            font_path = SCRIPT_DIR / font_path
        self.font_path_var.set(str(font_path))
        self.font_color_var.set(str(settings["font_color"]))
        self.top_font_size_var.set(str(settings["top_font_size"]))
        self.bottom_font_size_var.set(str(settings["bottom_font_size"]))
        self.text_margin_var.set(str(settings["text_margin"]))
        self.top_spacing_var.set(str(settings["top_spacing"]))
        self.bottom_spacing_var.set(str(settings["bottom_spacing"]))
        self.ffmpeg_var.set(str(settings["ffmpeg"]))
        self.ffprobe_var.set(str(settings["ffprobe"]))
        self.workers_var.set(str(settings["workers"]))
        self.skip_existing_var.set(bool(settings["skip_existing"]))
        self.debug_var.set(bool(settings["debug"]))

        self.top_text.delete("1.0", "end")
        self.top_text.insert(
            "1.0",
            str(settings.get("top_text", _read_optional_text(SCRIPT_DIR / "top_text.txt"))),
        )
        self.bottom_text.delete("1.0", "end")
        self.bottom_text.insert(
            "1.0",
            str(
                settings.get(
                    "bottom_text",
                    _read_optional_text(SCRIPT_DIR / "bottom_text.txt"),
                )
            ),
        )
        for target_mode in MODE_LABELS:
            self._refresh_targets(target_mode)
        self._switch_mode()

    def start_job(self):
        if self.running:
            return
        try:
            mode, config = self._build_config()
        except (OSError, ValueError) as exc:
            messagebox.showerror("設定無效", str(exc), parent=self.root)
            return

        self.running = True
        self._set_footer_buttons("disabled")
        self.status_var.set("正在偵測硬體並處理影片…")
        self._append_log("\n" + "=" * 60 + "\n")
        self._append_log(f"開始：{MODE_LABELS[mode]}\n")
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(mode, config),
            name="video-job",
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, mode, config):
        stream = QueueStream(self.event_queue)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                processor = VideoReframer(config) if mode == "reframe" else VideoCompressor(config)
                result = processor.run()
        except BaseException as exc:
            self.event_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        else:
            self.event_queue.put(("done", result))

    def _drain_events(self):
        try:
            while True:
                event, payload = self.event_queue.get_nowait()
                if event == "log":
                    self._append_log(payload)
                elif event == "error":
                    self._finish_with_error(payload)
                elif event == "done":
                    self._finish_success(payload)
                elif event == "update_check_done":
                    self._finish_update_check(payload)
                elif event == "update_check_error":
                    self._finish_update_check_error(payload)
                elif event == "update_progress":
                    received, total = payload
                    percent = int(received * 100 / total) if total else 0
                    self.update_status_var.set(
                        f"正在下載並驗證更新檔… {percent}% "
                        f"({received / 1024 / 1024:.1f} / "
                        f"{total / 1024 / 1024:.1f} MiB)"
                    )
                elif event == "update_staged":
                    self._finish_update_staged(payload)
                elif event == "update_install_error":
                    self._finish_update_install_error(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1500:
            self.log_text.delete("1.0", f"{line_count - 1200}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_footer_buttons(self, state):
        self.start_button.configure(state=state)
        self.save_button.configure(state=state)
        self.restore_button.configure(state=state)

    def _finish_success(self, result):
        self.running = False
        self._set_footer_buttons("normal")
        if self.update_info is not None and self.update_info.available:
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if allowed and not self.update_busy:
                self.install_update_button.configure(state="normal")
        success_count, failed_files = result
        if failed_files:
            self.status_var.set(f"完成：成功 {success_count}，失敗 {len(failed_files)}")
            messagebox.showwarning(
                "處理完成",
                f"成功 {success_count} 個；失敗：{', '.join(failed_files)}",
                parent=self.root,
            )
        else:
            self.status_var.set(f"完成：成功 {success_count} 個")
            messagebox.showinfo(
                "處理完成", f"成功處理 {success_count} 個影片。", parent=self.root
            )

    def _finish_with_error(self, error):
        self.running = False
        self._set_footer_buttons("normal")
        if self.update_info is not None and self.update_info.available:
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if allowed and not self.update_busy:
                self.install_update_button.configure(state="normal")
        self.status_var.set("處理失敗")
        self._append_log(f"\n[錯誤] {error}\n")
        messagebox.showerror("處理失敗", error, parent=self.root)

    def _on_close(self):
        if self.installing_update:
            self.root.destroy()
            return
        if self.update_busy:
            messagebox.showwarning(
                "更新進行中",
                "正在檢查或準備軟體更新，請稍候。",
                parent=self.root,
            )
            return
        if self.running:
            messagebox.showwarning(
                "工作進行中",
                "目前仍在處理影片。請等待工作完成後再關閉視窗，以免留下未完成的 FFmpeg 程序。",
                parent=self.root,
            )
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        AutoReframeGUI(root)
    except (ConfigStoreError, OSError, ValueError) as exc:
        messagebox.showerror("無法啟動", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
