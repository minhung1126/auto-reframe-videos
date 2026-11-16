"""Shared desktop presentation helpers."""

from pathlib import Path
from auto_reframe_core.reframe import VideoReframer
from auto_reframe_core.gui_options import CODEC_LABELS, RESOLUTION_LABELS
from auto_reframe_core.output_plans import output_root_for
from auto_reframe_core.video_utils import VideoProgressEvent
import tkinter as tk
from tkinter import ttk
from auto_reframe_core.gui_constants import (
    MODE_LABELS,
    OUTPUT_CONFLICT_CANCEL,
    OUTPUT_CONFLICT_DELETE,
    OUTPUT_CONFLICT_LABELS,
    OUTPUT_CONFLICT_OVERWRITE,
    OUTPUT_CONFLICT_SKIP,
)


class _ScrollableDialog:
    """Keep long messages scrollable and actions outside the scrolling area."""

    def __init__(self, parent, title: str, message: str):
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.title(title)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)
        self.window.bind("<Escape>", lambda _event: self._cancel())
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        body = ttk.Frame(self.window, padding=(16, 16, 16, 0))
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.message = tk.Text(
            body, wrap="word", width=76, height=20, font="TkDefaultFont",
            padx=12, pady=12,
        )
        self.message.insert("1.0", message)
        self.message.configure(state="disabled")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.message.yview)
        self.message.configure(yscrollcommand=scrollbar.set)
        self.message.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.message.bind("<Tab>", self._focus_next)
        self.message.bind("<Shift-Tab>", self._focus_previous)
        self.message.bind("<ISO_Left_Tab>", self._focus_previous)

        self.buttons = ttk.Frame(self.window, padding=16)
        self.buttons.grid(row=1, column=0, sticky="e")

    @staticmethod
    def _focus_next(event):
        event.widget.tk_focusNext().focus_set()
        return "break"

    @staticmethod
    def _focus_previous(event):
        event.widget.tk_focusPrev().focus_set()
        return "break"

    def _show(self, parent, default_button):
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        max_width = max(1, screen_width - 48)
        max_height = max(1, screen_height - 96)
        width = min(self.window.winfo_reqwidth(), max_width)
        height = min(self.window.winfo_reqheight(), max_height)
        self.window.minsize(
            min(width, max(360, self.buttons.winfo_reqwidth())),
            min(height, self.buttons.winfo_reqheight() + 120),
        )
        self.window.maxsize(max_width, max_height)
        x = min(
            max(0, parent.winfo_rootx() + (parent.winfo_width() - width) // 2),
            max(0, screen_width - width - 24),
        )
        y = min(
            max(0, parent.winfo_rooty() + (parent.winfo_height() - height) // 2),
            max(0, screen_height - height - 64),
        )
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.deiconify()
        default_button.configure(default="active")
        default_button.focus_set()
        self.window.bind("<Return>", lambda _event: self._invoke_focused(default_button))
        self.window.grab_set()

    def _invoke_focused(self, default_button):
        focused = self.window.focus_get()
        if focused not in self.buttons.winfo_children():
            focused = default_button
        focused.invoke()
        return "break"


class JobConfirmationDialog(_ScrollableDialog):
    """Modal job summary with an always-visible start button."""

    def __init__(self, parent, message: str):
        self.result = False
        super().__init__(parent, "確認開始處理", message)
        start_button = ttk.Button(
            self.buttons, text="開始處理", command=lambda: self._choose(True),
        )
        start_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(self.buttons, text="取消", command=self._cancel).grid(row=0, column=1)
        self._show(parent, start_button)

    def _choose(self, result: bool):
        self.result = result
        self.window.destroy()

    def _cancel(self):
        self._choose(False)

    @classmethod
    def ask(cls, parent, message: str) -> bool:
        dialog = cls(parent, message)
        parent.wait_window(dialog.window)
        return dialog.result


class OutputConflictDialog(_ScrollableDialog):
    """Modal four-choice dialog for existing target output folders."""

    def __init__(self, parent, conflicts: list[Path], prefer_skip: bool):
        self.result = OUTPUT_CONFLICT_CANCEL
        super().__init__(parent, "已有相同輸出目標", build_output_conflict_message(conflicts))
        buttons = self.buttons
        skip_button = ttk.Button(
            buttons,
            text="略過既有檔",
            command=lambda: self._choose(OUTPUT_CONFLICT_SKIP),
        )
        skip_button.grid(row=0, column=0, padx=(0, 8))
        overwrite_button = ttk.Button(
            buttons,
            text="覆寫同名檔",
            command=lambda: self._choose(OUTPUT_CONFLICT_OVERWRITE),
        )
        overwrite_button.grid(row=0, column=1)
        delete_button = ttk.Button(
            buttons,
            text="刪除本次輸出檔",
            command=lambda: self._choose(OUTPUT_CONFLICT_DELETE),
        )
        delete_button.grid(row=1, column=0, padx=(0, 8), pady=(8, 0))
        ttk.Button(buttons, text="取消", command=self._cancel).grid(row=1, column=1, pady=(8, 0))

        default_button = skip_button if prefer_skip else overwrite_button
        self._show(parent, default_button)

    def _choose(self, result: str):
        self.result = result
        self.window.destroy()

    def _cancel(self):
        self._choose(OUTPUT_CONFLICT_CANCEL)

    @classmethod
    def ask(cls, parent, conflicts: list[Path], prefer_skip: bool) -> str:
        dialog = cls(parent, conflicts, prefer_skip)
        parent.wait_window(dialog.window)
        return dialog.result


def build_job_confirmation_message(
    mode: str,
    config,
    output_conflict_action: str | None = None,
) -> str:
    """Describe the exact GUI job and ask the user to approve it."""
    watermark_file = Path(config.watermark_file).name
    watermark = f"已套用（{watermark_file}）" if config.watermark_enabled else "未套用"
    lines = [
        "請確認以下設定是否正確。",
        "",
        f"模式：{MODE_LABELS[mode]}",
        f"浮水印：{watermark}",
    ]
    if getattr(config, "video_files", None) is not None:
        lines.append(f"影片數量：{len(config.video_files)}")
        lines.append(
            "輸出模式："
            + (
                "各影片所在資料夾的 auto-reframe/"
                if config.output_mode == "source"
                else "指定資料夾"
            )
        )
        lines.append("實際目的地：")
        prepared = getattr(config, "prepared_batch", None)
        destinations = (
            prepared.destinations
            if prepared is not None
            else sorted({output_root_for(config, p) for p in config.video_files})
        )
        lines.extend(f"  {path}" for path in destinations)
    if output_conflict_action:
        lines.append(f"既有輸出：{OUTPUT_CONFLICT_LABELS[output_conflict_action]}")
    prepared = getattr(config, "prepared_batch", None)
    if prepared is not None and hasattr(prepared, "entries"):
        lines.append("實際輸出尺寸：")
        for entry in prepared.entries.values():
            for _, plan in entry.plans:
                for output in plan.active_maps:
                    lines.append(
                        f"  {entry.source.name}：{output.width} × {output.height}，{output.codec}"
                    )
            if entry.info.get("hdr") or entry.info.get("bit_depth", 8) > 8:
                lines.append(
                    "  此來源含 HDR／10-bit；目前輸出為 8-bit，尚未提供 HDR 保留或色調映射。"
                )
        lines.append("磁碟空間依位元率與備份估算；實際大小會隨媒體內容變化。")
    lines.extend(["", "輸出組合："])

    if mode == "reframe":
        final_width, final_height = config.final_ratio
        for index, target in enumerate(config.targets, 1):
            ratio_width, ratio_height = target["ratio"]
            lines.append(
                f"  {index}. 裁切 {ratio_width}:{ratio_height} → 畫布 "
                f"{final_width}:{final_height}／{RESOLUTION_LABELS[target['resolution']]}／"
                f"{CODEC_LABELS[target['vcodec']]}"
            )
        lines.extend(
            [
                "",
                "偵測到的上方文字：",
                VideoReframer._normalize_text(config.top_text_override or "")
                or "（未設定）",
                "",
                "偵測到的下方文字：",
                VideoReframer._normalize_text(config.bottom_text_override or "")
                or "（未設定）",
            ]
        )
    else:
        for index, target in enumerate(config.targets, 1):
            lines.append(
                f"  {index}. {RESOLUTION_LABELS[target['resolution']]}／"
                f"{CODEC_LABELS[target['vcodec']]}"
            )

    lines.extend(["", "要繼續開始處理嗎？"])
    return "\n".join(lines)


def build_output_conflict_message(conflicts: list[Path]) -> str:
    """Explain the exact output scope considered by the preflight check."""
    shown = [f"  • {path}" for path in conflicts[:8]]
    if len(conflicts) > len(shown):
        shown.append(f"  • ……另有 {len(conflicts) - len(shown)} 個")
    return "\n".join(
        [
            f"偵測到 {len(conflicts)} 個本次規劃中已存在的輸出檔案：",
            "",
            *shown,
            "",
            "只會處理上列檔案；其他影片與檔案不受影響。",
            "",
            "略過既有檔：保留全部內容，只產生缺少的同名輸出。",
            "覆寫同名檔：重新產生本次同名影片，保留資料夾內其他檔案。",
            "刪除本次輸出檔：刪除上列檔案（無法復原），再重新處理。",
            "取消：不開始本次工作。",
        ]
    )


def format_video_event_log(event: VideoProgressEvent) -> str:
    """Format only lifecycle events that should remain in the GUI history."""

    prefix = f"[{event.index}/{event.total}] {event.filename}"
    if event.kind == "started":
        return f"{prefix}：開始處理"
    if event.kind == "retry":
        detail = event.message or "重新嘗試"
        return f"{prefix}：{detail}（第 {event.attempt} 次嘗試）"
    if event.kind == "completed":
        return f"{prefix}：處理完成"
    if event.kind == "failed":
        detail = event.message or "處理未完成"
        return f"{prefix}：失敗－{detail}"
    if event.kind == "skipped":
        return f"{prefix}：既有輸出已略過"
    if event.kind == "cancelled":
        return f"{prefix}：已取消，已完成輸出保留"
    if event.kind == "stalled":
        return f"{prefix}：{event.message}"
    return ""
