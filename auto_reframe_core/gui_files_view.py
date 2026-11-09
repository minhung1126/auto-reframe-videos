"""Desktop files presentation and interaction."""

import tkinter as tk
from tkinter import filedialog, ttk
from auto_reframe_core.gui_constants import VIDEO_EXTENSIONS


class FilesView:
    def _build_files_tab(self):
        tab = self.files_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew", pady=8)
        for label, command in (
            ("新增影片", self._add_videos),
            ("新增資料夾", self._add_folder),
            ("移除選取", self._remove_videos),
            ("清空清單", self._clear_videos),
        ):
            ttk.Button(controls, text=label, command=command).pack(side="left", padx=4)
        ttk.Checkbutton(
            controls, text="包含子資料夾", variable=self.recursive_var
        ).pack(side="left", padx=8)
        self.files_tree = ttk.Treeview(
            tab, columns=("name", "source"), show="headings", selectmode="extended"
        )
        self.files_tree.heading("name", text="檔名")
        self.files_tree.heading("source", text="來源位置")
        self.files_tree.column("name", width=220)
        self.files_tree.column("source", width=600)
        self.files_tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.files_tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.files_tree.configure(yscrollcommand=scroll.set)
        ttk.Label(tab, textvariable=self.video_count_var).grid(
            row=2, column=0, sticky="w", pady=8
        )
        output = ttk.LabelFrame(tab, text="輸出位置", padding=10)
        output.grid(row=3, column=0, sticky="ew")
        output.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            output,
            text="各影片所在資料夾的 auto-reframe/（預設）",
            variable=self.output_mode_var,
            value="source",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Radiobutton(
            output, text="指定資料夾", variable=self.output_mode_var, value="specified"
        ).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(output, textvariable=self.output_folder_var).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(output, text="選擇…", command=self._choose_output).grid(
            row=1, column=2
        )
        ttk.Label(output, text="子資料夾（選填）").grid(
            row=2, column=0, sticky="w", pady=4
        )
        ttk.Entry(output, textvariable=self.output_subfolder_var).grid(
            row=2, column=1, sticky="ew", padx=8
        )
        ttk.Label(output, text="留空則直接使用上方輸出位置").grid(
            row=3, column=1, sticky="w", padx=8
        )
        ttk.Label(
            tab,
            text="可拖曳多個影片或資料夾進視窗。處理期間無法修改清單。原始影片不會搬移。",
        ).grid(row=4, column=0, sticky="w", pady=8)
        try:
            from tkinterdnd2 import DND_FILES

            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._drop_videos)
        except (ImportError, tk.TclError, AttributeError, RuntimeError):
            ttk.Label(
                tab,
                text="拖曳無法使用；請使用「新增影片／新增資料夾」。",
                foreground="#b00020",
            ).grid(row=5, column=0, sticky="w", pady=4)

    def _refresh_files(self):
        self.files_tree.delete(*self.files_tree.get_children())
        for index, path in enumerate(self.video_files):
            self.files_tree.insert(
                "", "end", iid=str(index), values=(path.name, str(path.parent))
            )
        self.video_count_var.set(f"影片總數：{len(self.video_files)}")

    def _drop_videos(self, event):
        self._add_items(self.root.tk.splitlist(event.data))
        return "copy"

    def _add_videos(self):
        from auto_reframe_core.video_list import VIDEO_EXTENSIONS

        files = filedialog.askopenfilenames(
            parent=self.root,
            title="新增影片",
            filetypes=(
                (
                    "影片",
                    " ".join("*" + extension for extension in sorted(VIDEO_EXTENSIONS)),
                ),
                ("所有檔案", "*"),
            ),
        )
        if files:
            self._add_items(files)

    def _add_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="新增資料夾")
        if folder:
            self._add_items((folder,))

    def _remove_videos(self):
        if self.running or self.scan_busy or self.update_busy:
            return
        selected = {int(item) for item in self.files_tree.selection()}
        self.video_files = tuple(
            path for index, path in enumerate(self.video_files) if index not in selected
        )
        self._refresh_files()

    def _clear_videos(self):
        if self.running or self.scan_busy or self.update_busy:
            return
        self.video_files = ()
        self._refresh_files()

    def _choose_output(self):
        folder = filedialog.askdirectory(parent=self.root, title="選擇輸出資料夾")
        if folder:
            self.output_folder_var.set(folder)
            self.output_mode_var.set("specified")
