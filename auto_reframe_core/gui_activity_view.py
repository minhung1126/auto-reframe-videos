"""Desktop activity presentation and interaction."""

from auto_reframe_core.video_utils import VideoProgressEvent
from tkinter import scrolledtext, ttk
from auto_reframe_core.gui_dialogs import format_video_event_log


class ActivityView:
    def _initialize_activity_sash(self, _event=None):
        """Set a portable initial split once the progress tab has real dimensions."""

        if self._activity_sash_initialized:
            return
        activity_height = self.activity_pane.winfo_height()
        if activity_height < 180:
            return
        split = max(100, min(round(activity_height * 0.58), activity_height - 100))
        self.activity_pane.sashpos(0, split)
        self._activity_sash_initialized = True

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1500:
            self.log_text.delete("1.0", f"{line_count - 1200}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reset_video_progress(self):
        for item in self.progress_tree.get_children():
            self.progress_tree.delete(item)
        self.video_progress_states.clear()
        self.progress_summary_var.set("正在建立影片任務…")

    def _handle_video_progress(self, event: VideoProgressEvent):
        item_id = f"video-{event.index}"
        current_phase = ""
        current_progress = ""
        if self.progress_tree.exists(item_id):
            values = self.progress_tree.item(item_id, "values")
            if len(values) >= 5:
                current_phase = values[3]
                current_progress = values[4]

        status_by_kind = {
            "queued": "排隊中",
            "started": "準備中",
            "progress": "重試中" if event.attempt > 1 else "處理中",
            "retry": "重試中",
            "completed": "已完成",
            "failed": "失敗",
            "skipped": "已略過",
            "cancelled": "已取消",
            "committing": "正在提交",
            "stalled": "等待回應",
        }
        status = status_by_kind.get(event.kind, "處理中")
        phase = event.phase or current_phase
        progress = (
            f"{max(0, min(100, event.progress))}%"
            if event.progress is not None
            else current_progress
        )
        values = (
            f"{event.index}/{event.total}",
            event.filename,
            status,
            phase,
            progress,
        )
        if self.progress_tree.exists(item_id):
            self.progress_tree.item(item_id, values=values)
        else:
            self.progress_tree.insert("", "end", iid=item_id, values=values)

        if event.kind == "queued":
            state = "queued"
        elif event.kind in {"completed", "skipped", "cancelled"}:
            state = event.kind
        elif event.kind == "failed":
            state = "failed"
        elif event.kind == "retry" or event.attempt > 1:
            state = "retry"
        else:
            state = "running"
        self.video_progress_states[event.index] = state
        self._update_progress_summary(event.total)

        log_line = format_video_event_log(event)
        if log_line:
            self._append_log(log_line + "\n")

    def _update_progress_summary(self, total: int):
        states = list(self.video_progress_states.values())
        completed = states.count("completed")
        failed = states.count("failed")
        active = states.count("running") + states.count("retry")
        queued = states.count("queued")
        parts = [
            f"總計 {total}",
            f"完成 {completed}",
            f"處理中 {active}",
            f"排隊 {queued}",
        ]
        if failed:
            parts.append(f"失敗 {failed}")
        parts.extend(
            (f"略過 {states.count('skipped')}", f"取消 {states.count('cancelled')}")
        )
        self.progress_summary_var.set("｜".join(parts))

    def _build_activity_tab(self):
        self.activity_tab.columnconfigure(0, weight=1)
        self.activity_tab.rowconfigure(0, weight=1)
        self.activity_pane = ttk.Panedwindow(self.activity_tab, orient="vertical")
        self.activity_pane.grid(row=0, column=0, sticky="nsew")
        self._activity_sash_initialized = False

        progress_frame = ttk.LabelFrame(
            self.activity_pane,
            text="影片處理進度",
            padding=8,
        )
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        ttk.Label(
            progress_frame,
            textvariable=self.progress_summary_var,
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        columns = ("sequence", "filename", "status", "phase", "progress")
        self.progress_tree = ttk.Treeview(
            progress_frame,
            columns=columns,
            show="headings",
            height=5,
        )
        headings = {
            "sequence": "序號",
            "filename": "影片",
            "status": "狀態",
            "phase": "目前階段",
            "progress": "目前進度",
        }
        widths = {
            "sequence": 70,
            "filename": 250,
            "status": 90,
            "phase": 310,
            "progress": 90,
        }
        for column in columns:
            self.progress_tree.heading(column, text=headings[column])
            self.progress_tree.column(
                column,
                width=widths[column],
                minwidth=60,
                stretch=column in {"filename", "phase"},
                anchor="w" if column in {"filename", "phase"} else "center",
            )

        progress_y_scroll = ttk.Scrollbar(
            progress_frame,
            orient="vertical",
            command=self.progress_tree.yview,
        )
        progress_x_scroll = ttk.Scrollbar(
            progress_frame,
            orient="horizontal",
            command=self.progress_tree.xview,
        )
        self.progress_tree.configure(
            yscrollcommand=progress_y_scroll.set,
            xscrollcommand=progress_x_scroll.set,
        )
        self.progress_tree.grid(row=1, column=0, sticky="nsew")
        progress_y_scroll.grid(row=1, column=1, sticky="ns")
        progress_x_scroll.grid(row=2, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(self.activity_pane, text="事件日誌", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=5,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 9),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.activity_pane.add(progress_frame, weight=3)
        self.activity_pane.add(log_frame, weight=2)
        self.activity_pane.bind("<Configure>", self._initialize_activity_sash)
