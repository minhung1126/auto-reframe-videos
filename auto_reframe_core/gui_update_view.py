"""Desktop update presentation and interaction."""

import os
import sys
import threading
import webbrowser
from copy import deepcopy
from pathlib import Path
from auto_reframe_core.config_store import ConfigStoreError
from auto_reframe_core.gui_settings import normalize_watermark_settings
from auto_reframe_core.updater import (
    GITHUB_OWNER,
    GITHUB_REPOSITORY,
    UpdateError,
    can_self_update,
    check_for_update,
    launch_installer,
    prepare_update,
)
from auto_reframe_core.runtime_paths import is_frozen
from auto_reframe_core.job_controller import JobState
from auto_reframe_core.version import __version__
from auto_reframe_core.platform_profile import open_directory
from tkinter import messagebox, scrolledtext, ttk
from auto_reframe_core.gui_constants import CONFIG_PATH, SCRIPT_DIR, UPDATE_ERROR_PATH


class UpdateView:
    def _build_update_tab(self):
        self.update_tab.columnconfigure(0, weight=1)
        about = ttk.LabelFrame(self.update_tab, text="Auto Reframe Videos", padding=12)
        about.grid(row=0, column=0, sticky="ew")
        about.columnconfigure(1, weight=1)
        ttk.Label(about, text="目前版本").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(about, text=f"v{__version__}").grid(
            row=0, column=1, sticky="w", pady=4
        )
        ttk.Label(about, text="更新來源").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=4
        )
        ttk.Label(
            about,
            text=f"github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases",
        ).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Button(about, text="開啟設定檔", command=self.open_config).grid(
            row=2, column=0, pady=8
        )
        ttk.Label(about, text=str(self.config_session.path), wraplength=650).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Label(
            about, text="請先關閉程式再編輯；外部修改於下次啟動載入。", wraplength=800
        ).grid(row=3, column=0, columnspan=2, sticky="w")
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
            text="下載安裝檔" if is_frozen() else "下載並安裝",
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
            "此安裝可自動更新；安裝時會備份舊版，使用者資料獨立保存。"
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

    def _open_directory(self, directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            open_directory(directory)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "無法開啟資料夾",
                f"{directory}\n{exc}",
                parent=self.root,
            )

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
        if self.update_busy or self.running or self.scan_busy:
            return
        self._cancel_preview()
        self._ensure_controller().begin(JobState.UPDATING)
        self.update_busy = True
        self.update_info = None
        self.check_update_button.configure(state="disabled")
        self.install_update_button.configure(state="disabled")
        self._set_footer_buttons("disabled")
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
            info = check_for_update(desktop=is_frozen())
        except BaseException as exc:
            self._emit_job("update_check_error", f"{type(exc).__name__}: {exc}")
        else:
            self._emit_job("update_check_done", info)

    def _finish_update_check(self, info):
        self._complete_controller()
        self.update_busy = False
        self._set_footer_buttons("normal")
        self.update_info = info
        self.check_update_button.configure(state="normal")
        immutable_text = "不可變 Release" if info.immutable else "一般 Release"
        if info.available:
            self.update_status_var.set(
                f"有新版本 v{info.latest_version}（目前 v{info.current_version}，"
                f"{immutable_text}）。"
            )
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if (allowed or is_frozen()) and not self.running and not self.scan_busy:
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
        self._complete_controller()
        self.update_busy = False
        self._set_footer_buttons("normal")
        self.check_update_button.configure(state="normal")
        self.install_update_button.configure(state="disabled")
        self.update_status_var.set("更新檢查失敗。")
        self.status_var.set("無法檢查更新")
        self._set_release_notes(error)
        messagebox.showerror("無法檢查更新", error, parent=self.root)

    def install_update(self):
        if self.scan_busy:
            return
        if is_frozen():
            if (
                self.running
                or self.update_busy
                or not self.update_info
                or not self.update_info.available
            ):
                return
            try:
                self.save_settings_for_upgrade()
                if not webbrowser.open(self.update_info.download_url):
                    raise webbrowser.Error(
                        "無法啟動瀏覽器，請使用 Release 頁面的下載連結。"
                    )
            except (ConfigStoreError, OSError, ValueError, webbrowser.Error) as exc:
                messagebox.showerror("無法下載更新", str(exc), parent=self.root)
                return
            messagebox.showinfo(
                "手動升級",
                "下載後請關閉程式，再安裝新版。設定與影片工作區會保留。",
                parent=self.root,
            )
            return
        if (
            self.update_busy
            or self.update_info is None
            or not self.update_info.available
        ):
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
            self._save_config(settings)
            self.settings = deepcopy(settings)
            self.watermark_settings = normalize_watermark_settings(self.settings)
        except (ConfigStoreError, OSError, ValueError) as exc:
            messagebox.showerror(
                "無法在更新前儲存設定",
                str(exc),
                parent=self.root,
            )
            return

        self._cancel_preview()
        self._ensure_controller().begin(JobState.UPDATING)
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
            self._emit_job("update_progress", (received, total))

        try:
            staged = prepare_update(info, SCRIPT_DIR, progress=report_progress)
        except BaseException as exc:
            self._emit_job("update_install_error", f"{type(exc).__name__}: {exc}")
        else:
            self._emit_job("update_staged", staged)

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
        self._complete_controller()
        self.update_busy = False
        self.check_update_button.configure(state="normal")
        if (
            self.update_info is not None
            and self.update_info.available
            and not self.running
            and not self.scan_busy
        ):
            self.install_update_button.configure(state="normal")
        self._set_footer_buttons("normal")
        self.update_status_var.set("更新安裝失敗。")
        self.status_var.set("更新安裝失敗")
        messagebox.showerror("無法安裝更新", error, parent=self.root)
