"""Explicit GUI qualification and comparable runtime evidence."""

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from auto_reframe_core.diagnostics import diagnostic_report
from auto_reframe_core.platform_profile import hidden_subprocess_kwargs
from auto_reframe_core.runtime_paths import tool_path, resource_root, is_frozen


def runtime_report(*, require_gui=False, root=None):
    report = diagnostic_report()
    report["gpu"] = "not measured"
    try:
        report["tkinterdnd2"] = version("tkinterdnd2")
    except PackageNotFoundError:
        report["tkinterdnd2"] = "bundled; see build record"
    build = resource_root() / "build-info.json"
    if build.is_file():
        report["commit"] = json.loads(build.read_text(encoding="utf-8")).get("commit")
    elif not is_frozen():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=resource_root(),
                capture_output=True,
                text=True,
                timeout=5,
                **hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                report["commit"] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        process = subprocess.run(
            [tool_path("ffmpeg"), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            **hidden_subprocess_kwargs(),
        )
        report["ffmpeg"] = (
            process.stdout.splitlines()[0] if process.returncode == 0 else "unavailable"
        )
    except (OSError, subprocess.TimeoutExpired):
        report["ffmpeg"] = "unavailable"
    owned = root is None
    try:
        if root is None:
            from auto_reframe_core.gui_style import create_gui_root

            root = create_gui_root(require_dnd=True)
            root.withdraw()
        report.update(
            tcl=str(root.tk.call("info", "patchlevel")),
            tk=str(root.tk.call("package", "require", "Tk")),
            tkdnd=str(root.tk.call("package", "require", "tkdnd")),
            scaling=float(root.tk.call("tk", "scaling")),
        )
        if not report["tk"].startswith("8.") or not report["tcl"].startswith("8."):
            raise RuntimeError("GUI 與 TkDND 必須使用 Tcl/Tk 8.x。")
        from tkinterdnd2 import DND_FILES

        root.drop_target_register(DND_FILES)
        root.update_idletasks()
    except Exception as error:
        report["gui_error"] = str(error)
        if require_gui:
            raise RuntimeError(f"GUI 環境驗證失敗：{error}") from error
    finally:
        if owned and root is not None:
            root.destroy()
    return report
