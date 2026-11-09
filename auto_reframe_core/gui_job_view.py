"""Desktop job presentation and interaction."""

import queue
import threading
import time
from auto_reframe_core.config_store import ConfigStoreError
from auto_reframe_core.updater import can_self_update
from auto_reframe_core.runtime_paths import is_frozen
from auto_reframe_core.job_models import JobEvent, JobRequest, BatchResult, ResultStatus
from auto_reframe_core.job_controller import JobController, JobState
from auto_reframe_core.batch_runner import BatchService
from auto_reframe_core.process_runner import TaskCancelled
from tkinter import messagebox
from auto_reframe_core.gui_constants import (
    MODE_LABELS,
    OUTPUT_CONFLICT_CANCEL,
    OUTPUT_CONFLICT_DELETE,
    OUTPUT_CONFLICT_SKIP,
    SCRIPT_DIR,
)
from auto_reframe_core.gui_dialogs import (
    OutputConflictDialog,
    build_job_confirmation_message,
)


class JobView:
    def _add_items(self, items):
        if self.running or self.update_busy:
            messagebox.showinfo(
                "處理中", "本次任務清單已固定，請完成後再修改。", parent=self.root
            )
            return
        if self.scan_busy:
            messagebox.showinfo(
                "正在掃描", "請等目前掃描完成後再加入影片。", parent=self.root
            )
            return
        self._cancel_preview()
        context = self._ensure_controller().begin(JobState.SCANNING)
        self.scan_busy = True
        self._set_footer_buttons("disabled")
        self.status_var.set("正在掃描影片…")
        self.scan_worker = threading.Thread(
            target=self._scan_items,
            args=(
                tuple(items),
                tuple(self.video_files),
                bool(self.recursive_var.get()),
            ),
            name="video-scan",
            daemon=True,
        )
        self.scan_worker.start()

    def _scan_items(self, items, existing, recursive):
        from auto_reframe_core.video_list import collect_videos

        try:
            result = collect_videos(
                items, existing, recursive, task_context=self.job_controller.context
            )
        except Exception as exc:
            self._emit_job(
                "cancelled"
                if self.job_controller.context.cancellation.cancelled
                else "scan_error",
                str(exc),
            )
        else:
            self._emit_job(
                "cancelled"
                if self.job_controller.context.cancellation.cancelled
                else "scan_done",
                result,
            )

    def _finish_scan(self, result):
        self._complete_controller()
        self.scan_busy = False
        self.video_files, errors = result
        self._refresh_files()
        self._set_footer_buttons("normal")
        self.status_var.set(f"掃描完成：{len(self.video_files)} 個影片")
        if errors:
            messagebox.showwarning(
                "部分項目無法加入", "\n".join(errors), parent=self.root
            )

    def _finish_scan_error(self, error):
        self._complete_controller()
        self.scan_busy = False
        self._set_footer_buttons("normal")
        self.status_var.set("影片掃描失敗")
        messagebox.showerror("無法掃描影片", error, parent=self.root)

    def start_job(self, *, request_override=None):
        if (
            self.running
            or self.scan_busy
            or self.update_busy
            or self._ensure_controller().busy
        ):
            return
        try:
            mode, config = (
                request_override
                if request_override is not None
                else self._build_config()
            )
        except (ConfigStoreError, OSError, ValueError) as exc:
            self._show_field_error(str(exc))
            messagebox.showerror("設定無效", str(exc), parent=self.root)
            return

        self._cancel_preview()
        context = self._ensure_controller().begin(JobState.PREPARING)
        self.last_request = JobRequest(mode, config, context.job_id)
        self.last_result = None
        if hasattr(self, "result_actions"):
            self.result_actions.grid_remove()
        if hasattr(self, "field_error_var"):
            self.field_error_var.set("")
        self.running = True
        self.preflight_busy = True
        self._set_footer_buttons("disabled")
        self.status_var.set("正在檢查影片與輸出位置…")
        self.worker = threading.Thread(
            target=self._prepare_job,
            args=(mode, config),
            name="video-preflight",
            daemon=True,
        )
        self.worker.start()

    def _prepare_job(self, mode, config):
        try:
            prepared = self._service().prepare(
                self.last_request, self.job_controller.context
            )
            conflicts = tuple(path for path in prepared.finals if path.exists())
        except Exception as exc:
            self.job_controller.context.release_outputs()
            self._emit_job(
                "cancelled"
                if self.job_controller.context.cancellation.cancelled
                else "preflight_error",
                str(exc),
            )
        else:
            self._emit_job("preflight_done", (mode, config, conflicts))

    def _cancel_preflight(self, status="已取消開始處理"):
        self._complete_controller()
        self.preflight_busy = False
        self.running = False
        self._set_footer_buttons("normal")
        self.status_var.set(status)

    def _finish_preflight_error(self, error):
        self._cancel_preflight("無法開始處理")
        messagebox.showerror("無法檢查影片與輸出", error, parent=self.root)

    def _finish_preflight(self, result):
        try:
            self._confirm_prepared_job(result)
        except Exception as exc:
            self._finish_preflight_error(str(exc))

    def _confirm_prepared_job(self, result):
        mode, config, conflicts = result
        controller = self._ensure_controller()
        if controller.state == JobState.CANCELLING:
            self._finish_cancelled()
            return
        if controller.state == JobState.PREPARING:
            controller.transition(JobState.CONFIRMING)

        conflict_action = None
        if conflicts:
            conflict_action = OutputConflictDialog.ask(
                self.root,
                conflicts,
                prefer_skip=config.skip_existing,
            )
            if conflict_action == OUTPUT_CONFLICT_CANCEL:
                self._cancel_preflight()
                return
            config.skip_existing = conflict_action == OUTPUT_CONFLICT_SKIP

        if not messagebox.askyesno(
            "確認開始處理",
            build_job_confirmation_message(mode, config, conflict_action),
            parent=self.root,
            default=messagebox.YES,
        ):
            self._cancel_preflight()
            return

        if controller.state == JobState.CONFIRMING:
            controller.transition(JobState.VALIDATING)
        self.status_var.set("正在確認來源與輸出狀態…")
        self.worker = threading.Thread(
            target=self._validate_job,
            args=(mode, config, conflict_action, conflicts),
            name="video-preflight-confirm",
            daemon=True,
        )
        self.worker.start()

    def _validate_job(self, mode, config, conflict_action, conflicts):
        try:
            self._service().reserve(
                self.last_request,
                self.job_controller.context,
                delete_conflicts=conflicts
                if conflict_action == OUTPUT_CONFLICT_DELETE
                else (),
            )
        except Exception as exc:
            self.job_controller.context.release_outputs()
            self._emit_job(
                "cancelled"
                if self.job_controller.context.cancellation.cancelled
                else "preflight_error",
                str(exc),
            )
        else:
            self._emit_job("job_ready", (mode, config))

    def _start_prepared_job(self, result):
        mode, config = result
        controller = self._ensure_controller()
        if controller.state == JobState.CANCELLING:
            controller.context.release_outputs()
            self._finish_cancelled()
            return
        if controller.state == JobState.VALIDATING:
            controller.transition(JobState.RUNNING)
        self.preflight_busy = False
        self.skip_existing_var.set(config.skip_existing)
        self.status_var.set("正在偵測硬體並處理影片…")
        self._reset_video_progress()
        self.notebook.select(self.activity_tab)
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
        context = self.job_controller.context
        try:
            result = self._service().execute(self.last_request, context)
        except TaskCancelled:
            context.release_outputs()
            self._emit_job("cancelled")
        except BaseException as error:
            context.release_outputs()
            self._emit_job(
                "cancelled" if context.cancellation.cancelled else "error",
                f"{type(error).__name__}: {error}",
            )
        else:
            self._emit_job("done", result)

    def _ensure_controller(self):
        if not hasattr(self, "job_controller"):
            self.job_controller = JobController(self.event_queue.put)
        return self.job_controller

    def _service(self):
        if not hasattr(self, "batch_service"):
            self.batch_service = BatchService()
        return self.batch_service

    def _emit_job(self, kind, payload=None):
        self.job_controller.context.emit(kind, payload)

    def _complete_controller(self):
        controller = getattr(self, "job_controller", None)
        if controller is not None and controller.busy:
            controller.finish()
        if controller is not None and controller.close_when_finished:
            self.root.after_idle(self.root.destroy)

    def stop_job(self, *, close=False):
        controller = self._ensure_controller()
        if controller.busy:
            controller.cancel(close=close)
            self.status_var.set("正在停止；已完成的輸出會保留…")
            if hasattr(self, "stop_button"):
                self.stop_button.configure(state="disabled")

    def _finish_cancelled(self, _payload=None):
        self.scan_busy = self.preflight_busy = self.running = False
        self._complete_controller()
        self._set_footer_buttons("normal")
        self.status_var.set("已取消；已完成的輸出會保留")

    def _drain_events(self):
        deadline = time.monotonic() + 0.008
        try:
            for _ in range(128):
                if time.monotonic() >= deadline:
                    break
                item = self.event_queue.get_nowait()
                if isinstance(item, JobEvent):
                    if item.kind.startswith("preview_"):
                        self._receive_preview(item)
                        continue
                    if not self._ensure_controller().accepts(item):
                        continue
                    event, payload = item.kind, item.payload
                else:
                    event, payload = item
                if event == "log":
                    self._append_log(payload)
                elif event == "video_progress":
                    self._handle_video_progress(payload)
                elif event == "error":
                    self._finish_with_error(payload)
                elif event == "done":
                    self._finish_success(payload)
                elif event == "cancelled":
                    self._finish_cancelled(payload)
                elif event == "encoders":
                    self._append_log(
                        "實際編碼器："
                        + "，".join(f"{key}: {value}" for key, value in payload.items())
                        + "\n"
                    )
                elif event == "scan_done":
                    self._finish_scan(payload)
                elif event == "scan_error":
                    self._finish_scan_error(payload)
                elif event == "preflight_done":
                    self._finish_preflight(payload)
                elif event == "preflight_error":
                    self._finish_preflight_error(payload)
                elif event == "job_ready":
                    self._start_prepared_job(payload)
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
        self._drain_after = self.root.after(20, self._drain_events)

    def _set_footer_buttons(self, state):
        if state == "normal" and (self.running or self.scan_busy or self.update_busy):
            state = "disabled"
        if hasattr(self, "stop_button"):
            controller = getattr(self, "job_controller", None)
            active = (
                controller is not None
                and controller.busy
                and controller.state not in {JobState.UPDATING, JobState.CANCELLING}
            )
            self.stop_button.configure(state="normal" if active else "disabled")
        if hasattr(self, "preview_button"):
            self.preview_button.configure(state=state)
        self.start_button.configure(state=state)
        self.save_button.configure(state=state)
        self.restore_button.configure(state=state)

    def _finish_success(self, result):
        self._complete_controller()
        if isinstance(result, BatchResult):
            self.last_result = result
            self.running = self.preflight_busy = False
            self._set_footer_buttons("normal")
            status = (
                "已取消"
                if result.status == ResultStatus.CANCELLED
                else "部分失敗"
                if result.failed_sources
                else "完成"
            )
            summary = (
                f"{status}：完成 {result.count(ResultStatus.SUCCESS)}；略過 {result.count(ResultStatus.SKIPPED)}；"
                f"失敗 {len(result.failed_sources)}；取消 {result.count(ResultStatus.CANCELLED)}；"
                f"已產生輸出 {result.completed_outputs}"
            )
            self.status_var.set(summary)
            self._append_log(summary + "\n")
            self._show_result_actions()
            return
        self.running = False
        self._set_footer_buttons("normal")
        if self.update_info is not None and self.update_info.available:
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if (allowed or is_frozen()) and not self.update_busy:
                self.install_update_button.configure(state="normal")
        success_count, failed_files = result
        if not self.video_progress_states:
            self.progress_summary_var.set("沒有可處理的影片")
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
        self._complete_controller()
        self.running = False
        self._set_footer_buttons("normal")
        if not self.video_progress_states:
            self.progress_summary_var.set("無法建立影片任務")
        if self.update_info is not None and self.update_info.available:
            allowed, _reason = can_self_update(SCRIPT_DIR)
            if (allowed or is_frozen()) and not self.update_busy:
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
        if self.scan_busy or self.preflight_busy or self.running:
            if messagebox.askyesno(
                "停止後關閉",
                "停止目前工作並在清理完成後關閉？已完成的輸出會保留。",
                parent=self.root,
            ):
                self.stop_job(close=True)
            return
        if getattr(self, "preview_service", None) is not None:
            self.preview_service.cancel(close=True)
            if self.preview_service.busy:
                self.root.after(50, self._on_close)
                return
        self.root.destroy()
