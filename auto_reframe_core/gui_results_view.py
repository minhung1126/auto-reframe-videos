"""Result actions preserve the original request and retry only failed sources."""

from dataclasses import replace
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk

from auto_reframe_core.diagnostics import export_diagnostics


class ResultsView:
    def _show_result_actions(self):
        if hasattr(self, "result_actions"):
            self.result_actions.grid()
        if not hasattr(self, "result_actions"):
            self.result_actions = ttk.Frame(self.activity_tab, padding=(0, 8))
            self.result_actions.grid(row=1, column=0, sticky="ew")
            for label, callback in (
                ("開啟輸出位置", self.open_result_outputs),
                ("查看原因", self.show_result_reasons),
                ("只重試失敗項目", self.retry_failed),
                ("匯出診斷摘要", self.export_result_diagnostics),
            ):
                ttk.Button(self.result_actions, text=label, command=callback).pack(
                    side="left", padx=4
                )

    def open_result_outputs(self):
        if self.last_request is None or self.last_request.config.prepared_batch is None:
            return
        destinations = self.last_request.config.prepared_batch.destinations
        if len(destinations) == 1:
            self._open_directory(destinations[0])
            return
        window = tk.Toplevel(self.root)
        window.title("輸出位置")
        listing = tk.Listbox(
            window, width=90, height=min(20, len(destinations)), exportselection=False
        )
        listing.pack(fill="both", expand=True, padx=10, pady=10)
        for destination in destinations:
            listing.insert("end", str(destination))

        def open_selected():
            for index in listing.curselection():
                self._open_directory(destinations[index])

        ttk.Button(window, text="開啟所選位置", command=open_selected).pack(pady=8)
        window.bind("<Escape>", lambda _: window.destroy())

    def show_result_reasons(self):
        if self.last_result is None:
            return
        window = tk.Toplevel(self.root)
        window.title("處理結果與原因")
        text = scrolledtext.ScrolledText(window, width=90, height=25, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        for video in self.last_result.videos:
            text.insert(
                "end", f"{video.source.name}：{video.status.value}\n{video.message}\n"
            )
            for target in video.targets:
                text.insert("end", f"  {target.path.name}：{target.status.value}\n")
        text.configure(state="disabled")
        window.bind("<Escape>", lambda _: window.destroy())

    def retry_failed(self):
        if (
            self.running
            or self.scan_busy
            or self.update_busy
            or self.last_result is None
        ):
            return
        failures = self.last_result.failed_sources
        selection = self.progress_tree.selection()
        if selection:
            indices = {int(item.removeprefix("video-")) - 1 for item in selection}
            selected = {
                video.source
                for index, video in enumerate(self.last_result.videos)
                if index in indices
            }
            failures = tuple(path for path in failures if path in selected)
        if not failures:
            messagebox.showinfo(
                "沒有失敗項目",
                "所選項目沒有失敗影片；取消項目不會自動重試。",
                parent=self.root,
            )
            return
        config = replace(
            self.last_request.config, video_files=failures, prepared_batch=None
        )
        self.start_job(request_override=(self.last_request.mode, config))

    def export_result_diagnostics(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="匯出診斷摘要",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
        )
        if path:
            export_diagnostics(path, self.last_result)
            self.status_var.set("已匯出環境與錯誤種類摘要；不含影片、文字或個人路徑")
