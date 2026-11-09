"""Shared retry, cancellation, and output lifecycle for both processing modes."""

from auto_reframe_core.output_plans import cleanup_temp_outputs, promote_temp_outputs
from auto_reframe_core.video_utils import emit_video_progress, h264, h265, tqdm_write
from auto_reframe_core.job_models import OutputTarget, ResultStatus


def build_ffmpeg_attempts(command_factory, active_maps, h264_encoder, h265_encoder):
    first_cmd = command_factory()
    attempts = [("硬體優先", first_cmd)]
    if "-hwaccel" in first_cmd:
        attempts.append(("停用硬體解碼、保留硬體編碼",
                         command_factory(disable_decode_hwaccel=True)))
    outputs = (OutputTarget(*entry) for entry in active_maps)
    if any((entry.codec == h264 and h264_encoder != "libx264")
           or (entry.codec == h265 and h265_encoder != "libx265") for entry in outputs):
        attempts.append(("軟體解碼與軟體編碼",
                         command_factory(disable_decode_hwaccel=True, force_software_encode=True)))
    return attempts


def execute_output_plan(plan, attempts, info, task_info, position_q, *, runner,
                        phase, description, progress_callback=None,
                        debug_log_path=None, before_promotion=None, task_context=None):
    """Run one output group and clean its temporary files on every exit path."""
    cancellation = task_context.cancellation if task_context is not None else getattr(position_q, "cancellation", None)

    def cancelled():
        return cancellation is not None and cancellation.cancelled

    try:
        if cancelled():
            return False
        returncode, stderr_log = 1, []
        for attempt_index, (attempt_label, cmd) in enumerate(attempts, 1):
            if cancelled():
                return False
            desc = description
            if attempt_index > 1:
                emit_video_progress(progress_callback, task_info, kind="retry", progress=0,
                                    phase=phase, attempt=attempt_index, message=attempt_label)
                if progress_callback is None:
                    tqdm_write(f"  [重試] {task_info[2].name}: {attempt_label}")
                desc += f" [重試 {attempt_index}]"
            ffmpeg_progress = None
            if progress_callback is not None:
                def ffmpeg_progress(percent, current_attempt=attempt_index, current_label=attempt_label):
                    emit_video_progress(progress_callback, task_info, kind="progress", progress=percent,
                                        phase=phase, attempt=current_attempt, message=current_label)
            returncode, stderr_log = runner(cmd, info, desc, position_q, debug_log_path,
                                           progress_callback=ffmpeg_progress,
                                           **({"task_context": task_context,
                                               "status_callback": lambda message: emit_video_progress(
                                                   progress_callback, task_info, kind="stalled", phase=phase, message=message)}
                                              if task_context is not None else {}))
            if cancelled():
                return False
            if returncode == 0:
                break
            cleanup_temp_outputs(plan.tmps)
        if returncode != 0:
            if task_context is not None:
                task_context.errors[task_info[2]] = ("encoding", "".join(stderr_log)[-3000:] or "FFmpeg 轉碼失敗")
            if progress_callback is None:
                tqdm_write(f" [失敗!] {description}")
            print(f"\n\n[FFmpeg Error] 處理影片 {task_info[2].name} 時失敗！")
            print(f"指令輸出結尾：\n{''.join(stderr_log)}")
            return False
        if before_promotion is not None:
            before_promotion()
        if cancelled():
            return False
        emit_video_progress(progress_callback, task_info, kind="committing", progress=100,
                            phase="正在確認／提交輸出")
        promote_temp_outputs(plan.tmps, plan.finals, cancellation=cancellation)
        if task_context is not None:
            task_context.record_targets(task_info[2], plan.finals, ResultStatus.SUCCESS)
        if progress_callback is None:
            tqdm_write(f"{description} 完成!")
        return True
    except InterruptedError:
        if cancelled():
            return False
        raise
    finally:
        cleanup_temp_outputs(plan.tmps)
