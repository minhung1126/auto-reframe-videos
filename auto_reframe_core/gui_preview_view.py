"""Preview controls; frame generation stays in the cancellable preview service."""

import base64
import math
import tkinter as tk
from tkinter import ttk

from auto_reframe_core.preview import PreviewService


class PreviewView:
    def open_preview(self):
        if self.running or self.scan_busy or self.update_busy:
            return
        if (
            getattr(self, "preview_window", None) is not None
            and self.preview_window.winfo_exists()
        ):
            self.preview_window.lift()
            self._schedule_preview()
            return
        window = self.preview_window = tk.Toplevel(self.root)
        window.title("單幀預覽")
        window.geometry("720x760")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        controls = ttk.Frame(window, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        self.preview_seconds = tk.StringVar(value="0")
        self.preview_target = tk.StringVar(value="1")
        ttk.Label(controls, text="時間（秒）").pack(side="left")
        ttk.Entry(controls, textvariable=self.preview_seconds, width=9).pack(
            side="left", padx=6
        )
        ttk.Label(controls, text="輸出目標序號").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.preview_target,
            width=5,
            state="readonly",
            values=[
                str(index)
                for index in range(1, len(self.targets[self._mode_key()]) + 1)
            ],
        ).pack(side="left", padx=6)
        ttk.Button(controls, text="更新預覽", command=self._schedule_preview).pack(
            side="left"
        )
        ttk.Button(controls, text="停止預覽", command=self._cancel_preview).pack(
            side="left", padx=6
        )
        self.preview_label = ttk.Label(window, anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_status = tk.StringVar(value="正在準備預覽…")
        ttk.Label(window, textvariable=self.preview_status, wraplength=680).grid(
            row=2, column=0, sticky="w", padx=10, pady=10
        )
        self.preview_service = PreviewService(self.event_queue.put)
        self.preview_seconds.trace_add("write", lambda *_: self._schedule_preview())
        self.preview_target.trace_add("write", lambda *_: self._schedule_preview())
        window.protocol("WM_DELETE_WINDOW", self._close_preview)
        window.bind("<Escape>", lambda _: self._close_preview())
        self._schedule_preview()

    def _schedule_preview(self, *_):
        if (
            not getattr(self, "preview_window", None)
            or not self.preview_window.winfo_exists()
        ):
            return
        if getattr(self, "preview_after", None):
            self.root.after_cancel(self.preview_after)
        self.preview_after = self.root.after(350, self._request_preview)

    def _request_preview(self):
        self.preview_after = None
        if self.running or self.scan_busy or self.update_busy:
            return
        try:
            mode, config = self._build_config()
            selected = self.files_tree.selection()
            source = (
                config.video_files[int(selected[0])]
                if selected
                else config.video_files[0]
            )
            target = int(self.preview_target.get()) - 1
            if not 0 <= target < len(config.targets):
                raise ValueError("請選擇目前模式的有效輸出目標序號。")
            self.preview_id = self.preview_service.request(
                mode, config, source, float(self.preview_seconds.get()), target
            )
            self.preview_status.set("正在擷取畫面…")
        except Exception as error:
            self.preview_status.set(str(error))

    def _receive_preview(self, event):
        if event.job_id != getattr(self, "preview_id", None):
            return
        if event.kind == "preview_error":
            self.preview_status.set(event.payload)
            return
        data, (width, height) = event.payload
        image = tk.PhotoImage(
            data=base64.b64encode(data).decode("ascii"),
            format="png",
            master=self.preview_window,
        )
        factor = max(1, math.ceil(max(width / 660, height / 630)))
        self.preview_image = image.subsample(factor)
        self.preview_label.configure(image=self.preview_image)
        self.preview_status.set(f"正式輸出尺寸 {width} × {height}；預覽已縮小顯示。")

    def _cancel_preview(self):
        if getattr(self, "preview_service", None):
            self.preview_service.cancel()
            self.preview_id = None
            if getattr(self, "preview_status", None):
                self.preview_status.set("已停止預覽")

    def _close_preview(self):
        if getattr(self, "preview_after", None):
            self.root.after_cancel(self.preview_after)
            self.preview_after = None
        if getattr(self, "preview_service", None):
            self.preview_service.cancel(close=True)
        self.preview_id = None
        self.preview_window.destroy()
        self.preview_window = None
