"""Desktop modes presentation and interaction."""

from pathlib import Path
from auto_reframe_core.gui_options import (
    CODEC_KEYS_BY_LABEL,
    CODEC_LABELS,
    CODEC_OPTIONS,
    POSITION_GRID_3X3,
    RATIO_OPTIONS,
    RESOLUTION_KEYS_BY_LABEL,
    RESOLUTION_LABELS,
    RESOLUTION_OPTIONS,
    WATERMARK_POSITION_LABELS,
    list_watermark_pngs,
    parse_ratio,
)
from auto_reframe_core.gui_settings import normalize_target_sets
from auto_reframe_core.runtime_paths import is_frozen
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from auto_reframe_core.gui_constants import (
    CREDIT_SYMBOL,
    MODE_KEYS_BY_LABEL,
    MODE_LABELS,
    WATERMARK_DIR,
)
from auto_reframe_core.gui_style import ScrollableTab, copy_text_to_clipboard


class ModesView:
    def _build_reframe_tab(self):
        self.reframe_tab.columnconfigure(0, weight=1)
        self.reframe_tab.rowconfigure(0, weight=1)
        self.reframe_notebook = ttk.Notebook(self.reframe_tab)
        self.reframe_notebook.grid(row=0, column=0, sticky="nsew")

        self.reframe_output_tab = ScrollableTab(self.reframe_notebook, padding=10)
        self.context_tab = ScrollableTab(self.reframe_notebook, padding=10)
        self.text_style_tab = ScrollableTab(self.reframe_notebook, padding=10)
        self.reframe_notebook.add(self.reframe_output_tab, text="輸出設定")
        self.reframe_notebook.add(self.context_tab, text="編輯上下文")
        self.reframe_notebook.add(self.text_style_tab, text="文字樣式")

        self._build_mode_output_tab(self.reframe_output_tab.content, "reframe")
        self._build_context_tab()
        self._build_text_style_tab()

    def _build_compress_tab(self):
        self.compress_tab.columnconfigure(0, weight=1)
        self.compress_tab.rowconfigure(0, weight=1)
        self.compress_notebook = ttk.Notebook(self.compress_tab)
        self.compress_notebook.grid(row=0, column=0, sticky="nsew")

        self.compress_output_tab = ScrollableTab(self.compress_notebook, padding=10)
        self.compress_notebook.add(self.compress_output_tab, text="輸出設定")
        self._build_mode_output_tab(self.compress_output_tab.content, "compress")

    def _build_mode_output_tab(self, parent, mode):
        parent.columnconfigure(0, weight=1)

        ttk.Label(
            parent, text="兩種模式共用〔待處理影片〕清單與輸出位置。直接處理原始檔。"
        ).grid(row=0, column=0, sticky="w")

        targets = ttk.LabelFrame(parent, text="輸出目標（可加入多個組合）", padding=10)
        targets.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        for column in range(3):
            targets.columnconfigure(column, weight=1)
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
        target_tree.column("ratio", width=120, minwidth=80, anchor="center")
        target_tree.column("resolution", width=210, minwidth=120, anchor="center")
        target_tree.column("codec", width=150, minwidth=100, anchor="center")
        target_tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
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

        ttk.Label(targets, text="解析度").grid(
            row=1, column=1, sticky="w", pady=(10, 0)
        )
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
        ).grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Button(
            targets,
            text="移除選取",
            command=lambda selected_mode=mode: self._remove_targets(selected_mode),
        ).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Button(
            targets,
            text="恢復預設",
            command=lambda selected_mode=mode: self._reset_targets(selected_mode),
        ).grid(row=3, column=2, sticky="ew", pady=(8, 0))

        watermark = ttk.LabelFrame(parent, text="PNG 浮水印", padding=10)
        watermark.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        watermark.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            watermark,
            text="蓋浮水印",
            variable=self.watermark_enabled_vars[mode],
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        watermark_combo = ttk.Combobox(
            watermark,
            textvariable=self.watermark_file_vars[mode],
            state="readonly",
        )
        self.watermark_combos[mode] = watermark_combo
        watermark_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(watermark, text="重新整理", command=self.refresh_watermarks).grid(
            row=0, column=2, padx=(8, 0)
        )

        ttk.Button(
            watermark,
            text="開啟資料夾",
            command=lambda: self._open_directory(self.watermark_dir),
        ).grid(row=0, column=3, padx=(8, 0))
        params = ttk.Frame(watermark)
        params.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        pos_box = ttk.Frame(params)
        pos_box.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 24))

        pos_header = ttk.Frame(pos_box)
        pos_header.grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(pos_header, text="位置：").pack(side="left")
        ttk.Label(
            pos_header,
            textvariable=self.watermark_position_vars[mode],
        ).pack(side="left")

        grid_frame = ttk.Frame(pos_box, style="AnchorGrid.TFrame", padding=3)
        grid_frame.grid(row=1, column=0, sticky="w")

        self.watermark_position_radios[mode] = {}
        for r, row in enumerate(POSITION_GRID_3X3):
            for c, pos_key in enumerate(row):
                pos_label = WATERMARK_POSITION_LABELS[pos_key]
                rb = ttk.Radiobutton(
                    grid_frame,
                    text="",
                    value=pos_label,
                    variable=self.watermark_position_vars[mode],
                    style="Compact.TRadiobutton",
                    cursor="hand2",
                )
                rb.grid(row=r, column=c, padx=0, pady=2)
                self.watermark_position_radios[mode][pos_key] = rb

        inputs_box = ttk.Frame(params)
        inputs_box.grid(row=0, column=1, sticky="w")

        ttk.Label(inputs_box, text="等比例").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(
            inputs_box,
            textvariable=self.watermark_width_ratio_vars[mode],
            width=8,
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(inputs_box, text="透明度").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(
            inputs_box,
            textvariable=self.watermark_opacity_vars[mode],
            width=8,
        ).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(inputs_box, text="邊距").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(
            inputs_box,
            textvariable=self.watermark_margin_vars[mode],
            width=8,
        ).grid(row=2, column=1, sticky="w", pady=2)

        hint_ratio = "0.15（裁切重製）" if mode == "reframe" else "0.10（影片壓縮）"
        ttk.Label(
            watermark,
            text=f"來源：{self.watermark_dir}；預設下方中央、等比例 {hint_ratio}、透明度 0.8（80%）、垂直插入 3。",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_context_tab(self):
        content = self.context_tab.content
        content.columnconfigure(0, weight=1)
        ttk.Label(
            content,
            text="最終畫布固定為 9:16，並永遠維持「上方文字／中央影片／下方文字」三層。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        credit = ttk.LabelFrame(
            content,
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

        ttk.Label(content, text="上方文字（可多行）").grid(row=2, column=0, sticky="w")
        self.top_text = scrolledtext.ScrolledText(
            content,
            height=5,
            wrap="word",
            font=self.editor_font,
        )
        self.top_text.grid(row=3, column=0, sticky="nsew", pady=(4, 10))

        ttk.Label(content, text="下方文字（可多行）").grid(row=4, column=0, sticky="w")
        self.bottom_text = scrolledtext.ScrolledText(
            content,
            height=4,
            wrap="word",
            font=self.editor_font,
        )
        self.bottom_text.grid(row=5, column=0, sticky="nsew", pady=(4, 0))
        content.rowconfigure(3, weight=1)
        content.rowconfigure(5, weight=1)

    def _build_text_style_tab(self):
        parent = self.text_style_tab.content
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="此頁設定只套用到「裁切重製」的上、下方文字。",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        style = ttk.LabelFrame(parent, text="文字樣式", padding=10)
        style.grid(row=1, column=0, sticky="ew")
        style.columnconfigure(1, weight=1)

        ttk.Label(style, text="字型").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=3
        )
        ttk.Entry(style, textvariable=self.font_path_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(style, text="瀏覽…", command=self._browse_font).grid(
            row=0, column=2, padx=(8, 0)
        )
        self._labeled_entry(style, "顏色", self.font_color_var, 1, 0)
        self._labeled_entry(style, "上方字級", self.top_font_size_var, 2, 0)
        self._labeled_entry(style, "下方字級", self.bottom_font_size_var, 3, 0)
        self._labeled_entry(style, "影片邊距", self.text_margin_var, 4, 0)
        self._labeled_entry(style, "上方行距", self.top_spacing_var, 5, 0)
        self._labeled_entry(style, "下方行距", self.bottom_spacing_var, 6, 0)

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

    def _advanced_path_row(self, label, variable, row):
        ttk.Label(self.advanced_tab, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Entry(
            self.advanced_tab,
            textvariable=variable,
            state="readonly" if is_frozen() else "normal",
        ).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(
            self.advanced_tab,
            text="選擇…",
            command=lambda target=variable: self._browse_executable(target),
            state="disabled" if is_frozen() else "normal",
        ).grid(row=row, column=2, padx=(8, 0), pady=5)

    def _load_initial_text(self):
        top_text = self.settings.get("top_text", "")
        bottom_text = self.settings.get("bottom_text", "")
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
            messagebox.showinfo(
                "目標已存在", "相同的輸出目標已在清單中。", parent=self.root
            )
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
            self.watermark_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "無法建立 watermark 資料夾", str(exc), parent=self.root
            )
            return

        files = list_watermark_pngs(self.watermark_dir)
        self.watermark_paths = {item.name: item for item in files}
        names = tuple(self.watermark_paths)
        for mode, watermark_combo in self.watermark_combos.items():
            watermark_combo.configure(
                values=names,
                state="readonly" if names else "disabled",
            )
            current = self.watermark_file_vars[mode].get()
            if current not in self.watermark_paths:
                self.watermark_file_vars[mode].set(names[0] if names else "")
        if not names:
            self.status_var.set("watermark/ 尚無 PNG；需要浮水印時請先放入圖片")
