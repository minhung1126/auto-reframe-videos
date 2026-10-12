"""Shared retry, cancellation, and output lifecycle for both processing modes."""

from auto_reframe_core.output_plans import cleanup_temp_outputs, promote_temp_outputs
from auto_reframe_core.video_utils import emit_video_progress, h264, h265, tqdm_write


def build_ffmpeg_attempts(command_factory, active_maps, h264_encoder, h265_encoder):
    first_cmd = command_factory()
    attempts = [("硬體優先", first_cmd)]
    if "-hwaccel" in first_cmd:
        attempts.append(("停用硬體解碼、保留硬體編碼",
                         command_factory(disable_decode_hwaccel=True)))
    if any((entry[5] == h264 and h264_encoder != "libx264")
           or (entry[5] == h265 and h265_encoder != "libx265") for entry in active_maps):
        attempts.append(("軟體解碼與軟體編碼",
                         command_factory(disable_decode_hwaccel=True, force_software_encode=True)))
    return attempts


def execute_output_plan(plan, attempts, info, task_info, position_q, *, runner,
                        phase, description, progress_callback=None,
                        debug_log_path=None, before_promotion=None):
    """Run one output group and clean its temporary files on every exit path."""
    cancellation = getattr(position_q, "cancellation", None)

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
                                           progress_callback=ffmpeg_progress)
            if cancelled():
                return False
            if returncode == 0:
                break
            cleanup_temp_outputs(plan.tmps)
        if returncode != 0:
            if progress_callback is None:
                tqdm_write(f" [失敗!] {description}")
            print(f"\n\n[FFmpeg Error] 處理影片 {task_info[2].name} 時失敗！")
            print(f"指令輸出結尾：\n{''.join(stderr_log)}")
            return False
        if before_promotion is not None:
            before_promotion()
        if cancelled():
            return False
        promote_temp_outputs(plan.tmps, plan.finals, cancellation=cancellation)
        if progress_callback is None:
            tqdm_write(f"{description} 完成!")
        return True
    except InterruptedError:
        if cancelled():
            return False
        raise
    finally:
        cleanup_temp_outputs(plan.tmps)
