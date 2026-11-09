"""Exercise real widgets and job service using only temporary synthetic media."""

from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

from auto_reframe_core.job_models import ResultStatus
from auto_reframe_core.runtime_report import runtime_report


def exercise_gui(source, temporary, *, timeout=90, root=None, progress=None):
    from tkinterdnd2 import TkinterDnD
    from auto_reframe_core.gui import AutoReframeGUI, MODE_LABELS

    record = progress or (lambda stage: None)
    owned_root = root is None
    record("GUI: initialize Tk")
    root = root if root is not None else TkinterDnD.Tk()
    root.withdraw()
    temporary = Path(temporary)
    try:
        record("GUI: qualify runtime")
        report = runtime_report(require_gui=True, root=root)
        record("GUI: construct application")
        app = AutoReframeGUI(root, data_root=temporary / "gui-data")
        app.output_mode_var.set("specified")
        app.output_folder_var.set(str(temporary / "gui-output"))
        app.output_subfolder_var.set("中文 子資料夾")

        def pump_until(predicate):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= deadline:
                    app.stop_job()
                    raise RuntimeError("完整 GUI 測試逾時。")
                root.update()
                time.sleep(0.01)
            root.update()

        record("GUI: scan original source")
        app._drop_videos(SimpleNamespace(data=root.tk.call("list", str(source))))
        pump_until(lambda: not app.scan_busy)
        if app.video_files != (Path(source).resolve(),):
            raise RuntimeError("GUI 原始影片清單或拖曳失敗。")
        outcomes = []
        with (
            patch("tkinter.messagebox.askyesno", return_value=True),
            patch("tkinter.messagebox.showinfo"),
            patch("tkinter.messagebox.showwarning"),
            patch("tkinter.messagebox.showerror") as errors,
        ):
            for mode in ("reframe", "compress"):
                record("GUI: " + mode)
                target = {"resolution": "source", "vcodec": "h264"}
                if mode == "reframe":
                    target["ratio"] = (4, 5)
                app.targets[mode] = [target]
                app.mode_var.set(MODE_LABELS[mode])
                app.top_text.delete("1.0", "end")
                app.top_text.insert("1.0", "中文 GUI 100%\n上方文字")
                app.bottom_text.insert("1.0", "下方文字")
                app.last_result = None
                app.start_button.invoke()
                pump_until(lambda: not app.running)
                if (
                    app.last_result is None
                    or app.last_result.count(ResultStatus.SUCCESS) != 1
                ):
                    raise RuntimeError(
                        f"GUI {mode} 工作沒有成功：{app.status_var.get()}"
                    )
                outcomes.append(mode)
            record("GUI: save settings")
            app.save_button.invoke()
            record("GUI: restore settings")
            app.restore_button.invoke()
            if errors.called:
                raise RuntimeError(f"GUI 顯示錯誤：{errors.call_args}")
        report["gui_operations"] = [
            "drop",
            "file-list",
            "output-subfolder",
            *outcomes,
            "save",
            "restore",
        ]
        return report
    finally:
        if owned_root:
            root.destroy()
