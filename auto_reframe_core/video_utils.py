"""Shared FFmpeg probing, progress, bitrate, and cancellation utilities."""

import json
import math
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from auto_reframe_core.process_runner import (
    ProcessRunner, ProcessCancellation, TaskCancelled, terminate_process,
)
from typing import Any, Callable, Dict, List, Optional, Tuple

from auto_reframe_core.encoder_profiles import (
    build_encoder_args,
    build_output_args,
    detect_h264_hw_encoder,
    detect_h265_hw_encoder,
    detect_hwaccel_for_cmd,
    double_bitrate,
)
from auto_reframe_core.platform_profile import hidden_subprocess_kwargs, resolve_workers

# FFmpeg autorotate selects transpose at ±0.5° from a quarter turn; outside
# that range it keeps the coded canvas and applies a general rotate filter.
_QUARTER_TURN_TOLERANCE = 0.5


@dataclass(frozen=True)
class VideoProgressEvent:
    """Structured state update for one input video in a parallel batch."""

    kind: str
    index: int
    total: int
    filename: str
    progress: Optional[int] = None
    phase: str = ""
    attempt: int = 1
    message: str = ""


VideoProgressCallback = Optional[Callable[[VideoProgressEvent], None]]


def emit_video_progress(
    callback: VideoProgressCallback,
    task_info,
    *,
    kind: str,
    progress: Optional[int] = None,
    phase: str = "",
    attempt: int = 1,
    message: str = "",
) -> None:
    """Deliver a task-scoped progress event when a structured sink is present."""

    if callback is None:
        return
    index, total, file_path = task_info
    callback(
        VideoProgressEvent(
            kind=kind,
            index=index,
            total=total,
            filename=file_path.name,
            progress=progress,
            phase=phase,
            attempt=attempt,
            message=message,
        )
    )


def parse_fps(fps_str: str) -> float:
    """安全解析 FFmpeg 的 fps 字串（如 '30000/1001' 或 '29.97'）"""
    try:
        if '/' in fps_str:
            num, den = fps_str.split('/')
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 30.0
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return 30.0


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_rotation(value: Any) -> Optional[float]:
    """將 FFprobe 的旋轉角度正規化為 [0, 360)，並吸附常見的直角誤差。"""
    try:
        rotation = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(rotation):
        return None

    rotation %= 360.0
    nearest_quarter_turn = (round(rotation / 90.0) * 90) % 360
    if min(
        abs(rotation - nearest_quarter_turn),
        abs(rotation - nearest_quarter_turn - 360),
        abs(rotation - nearest_quarter_turn + 360),
    ) <= _QUARTER_TURN_TOLERANCE:
        return float(nearest_quarter_turn)
    return rotation


def _display_matrix_rotation(value: Any) -> Optional[float]:
    """Recover the non-rounded angle from FFprobe's textual display matrix."""
    if not isinstance(value, str):
        return None
    rows = []
    for line in value.splitlines():
        if ":" not in line:
            continue
        _offset, values = line.split(":", 1)
        numbers = re.findall(r"-?\d+", values)
        if len(numbers) >= 3:
            rows.append(tuple(int(number) for number in numbers[:3]))
    if len(rows) < 2:
        return None
    matrix_a = rows[0][0]
    matrix_c = rows[1][0]
    if matrix_a == 0 and matrix_c == 0:
        return None
    return _normalize_rotation(math.degrees(math.atan2(matrix_c, matrix_a)))


def _stream_rotation(v_stream: Dict[str, Any]) -> float:
    """優先讀取 display matrix side data，否則回退至舊式 rotate tag。"""
    side_data_list = v_stream.get("side_data_list")
    if isinstance(side_data_list, list):
        for side_data in side_data_list:
            if not isinstance(side_data, dict):
                continue
            rotation = _display_matrix_rotation(side_data.get("displaymatrix"))
            if rotation is not None:
                return rotation
            if "rotation" in side_data:
                rotation = _normalize_rotation(side_data.get("rotation"))
                if rotation is not None:
                    return rotation

    tags = v_stream.get("tags")
    if isinstance(tags, dict):
        for key, value in tags.items():
            if str(key).casefold() == "rotate":
                rotation = _normalize_rotation(value)
                if rotation is not None:
                    return rotation

    return 0.0


def get_video_info(ffprobe_path: str, input_file: Path, *, task_context=None) -> Optional[Dict[str, Any]]:
    """呼叫 FFprobe 獲取影片資訊；width/height 為 FFmpeg autorotate 後的顯示尺寸。"""
    try:
        res = ProcessRunner(task_context).capture(
            [ffprobe_path, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(input_file)],
            timeout=30,
            **hidden_subprocess_kwargs(),
        )
        if res.returncode != 0 or not res.stdout.strip():
            print(f"  [警告] FFprobe 無法解析影片: {input_file.name}")
            return None
        data = json.loads(res.stdout)
    except TaskCancelled:
        raise
    except Exception as e:
        if task_context is not None:
            task_context.errors[Path(input_file)] = (
                "probe_timeout" if isinstance(e, subprocess.TimeoutExpired) else "probe",
                f"FFprobe 無法讀取影片：{e}",
            )
            task_context.emit("log", f"FFprobe：{type(e).__name__}: {e}")
        print(f"  [警告] 無法取得影片資訊: {input_file.name} ({e})")
        return None

    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    if not v_stream:
        print(f"  [警告] 影片 {input_file.name} 中未找到視訊串流，已跳過。")
        return None

    source_width = _safe_int(v_stream.get("width"), 0)
    source_height = _safe_int(v_stream.get("height"), 0)
    if source_width <= 0 or source_height <= 0:
        print(
            f"  [警告] 影片 {input_file.name} 解析度異常"
            f"（{source_width}x{source_height}），已跳過。"
        )
        return None

    coded_width = _safe_int(v_stream.get("coded_width"), source_width)
    coded_height = _safe_int(v_stream.get("coded_height"), source_height)
    rotation = _stream_rotation(v_stream)
    swaps_dimensions = rotation in (90.0, 270.0)
    width, height = (
        (source_height, source_width)
        if swaps_dimensions
        else (source_width, source_height)
    )

    fps_str = v_stream.get("r_frame_rate", "30/1")
    fps = parse_fps(fps_str)

    duration = _safe_float(data.get("format", {}).get("duration", 0), 0.0)

    return {
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "coded_width": coded_width,
        "coded_height": coded_height,
        "rotation": int(rotation) if rotation.is_integer() else rotation,
        "fps": round(fps, 3),
        "duration": max(duration, 0.0),
        "has_audio": a_stream is not None,
        "pix_fmt": v_stream.get("pix_fmt", ""),
        "color_transfer": v_stream.get("color_transfer", ""),
        "color_primaries": v_stream.get("color_primaries", ""),
        "hdr": v_stream.get("color_transfer") in {"smpte2084", "arib-std-b67"},
        "bit_depth": _safe_int(v_stream.get("bits_per_raw_sample"), 10 if "10" in v_stream.get("pix_fmt", "") else 8),
    }


def get_youtube_bitrate(short_side: int, fps: float, multiplier: float = 1.5) -> str:
    """根據 YouTube 標準建議與影像短邊高度決定 bitrate，並乘上自訂倍率"""
    high_fps = fps > 30

    # 定義基底 Bitrate (以 Mbps 為單位)
    if short_side >= 2160:
        base_mbps = 60.0 if high_fps else 40.0
    elif short_side >= 1440:
        base_mbps = 24.0 if high_fps else 16.0
    elif short_side >= 1080:
        base_mbps = 12.0 if high_fps else 8.0
    elif short_side >= 720:
        base_mbps = 7.5 if high_fps else 5.0
    elif short_side >= 480:
        base_mbps = 4.0 if high_fps else 2.5
    else:
        base_mbps = 1.5 if high_fps else 1.0

    # 乘上倍率
    final_mbps = base_mbps * multiplier

    # 如果大於等於 1 Mbps，回傳 M；若小於 1 Mbps 則換算成 K
    if final_mbps >= 1.0:
        # 優化：如果是整數則轉成 int 避免 .0，如果是小數則保留 (FFmpeg 支援如 11.25M)
        val_str = f"{int(final_mbps)}" if final_mbps.is_integer() else f"{final_mbps}"
        return f"{val_str}M"
    else:
        return f"{int(final_mbps * 1000)}K"


def parse_ffmpeg_time(time_str: str) -> float:
    """將 FFmpeg 的 HH:MM:SS.ms 時間字串解析為秒數"""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0

def cleanup_tmp_files(out_dir: Path):
    """清理先前執行殘留的 .tmp 暫存檔"""
    if not out_dir.exists():
        return
    tmp_files = list(out_dir.rglob("*.tmp"))
    if tmp_files:
        deleted_count = 0
        for tmp in tmp_files:
            try:
                # 只有當暫存檔修改時間超過 10 分鐘以上，才判定為先前殘留並清理，避免誤刪平行執行的暫存檔
                if time.time() - tmp.stat().st_mtime > 600:
                    tmp.unlink()
                    deleted_count += 1
            except OSError:
                pass
        if deleted_count > 0:
            print(f"  [清理] 發現並清理了 {deleted_count} 個殘留暫存檔。")


# =============================================================================
# 以下為 Compress 與 Reframe 模式共用的常數與工具函式
# =============================================================================

# 方便在設定中使用，不用加引號
h264 = "h264"
h265 = "h265"

# 解析度關鍵字對應表 (指定短邊長度)
RESOLUTION_MAP: Dict[str, int] = {
    "4k":     2160,
    "2k":     1440,
    "1080p":  1080,
    "fhd":    1080,
    "720p":   720,
    "hd":     720,
    "480p":   480,
    "360p":   360,
    "source": 99999,  # 不縮放，直接取原始短邊
}

# tqdm 可用性（模組層級，只偵測一次）
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _tqdm = None
    _HAS_TQDM = False


def resolve_short_side(res_key: str, source_short: int) -> int:
    """給定解析度關鍵字與來源短邊，回傳最終輸出短邊；source 保留原始短邊，其餘不放大並向下取標準值。"""
    key = res_key.lower()
    if key == "source":
        return source_short

    target_limit = RESOLUTION_MAP.get(key, 1080)
    upper = min(target_limit, source_short)
    for s in (2160, 1440, 1080, 720, 480, 360):
        if s <= upper:
            return s
    return upper


def resolution_label(final_short: int) -> str:
    """將短邊像素值轉為人類可讀的解析度標籤（例如 1080 → 'FHD'）。"""
    if final_short >= 2160: return "4K"
    if final_short >= 1440: return "2K"
    if final_short >= 1080: return "FHD"
    if final_short >= 720:  return "HD"
    return f"{final_short}P"


def tqdm_write(msg: str) -> None:
    """使用 tqdm.write（若已安裝）或 print 輸出一行訊息，避免干擾進度條顯示。"""
    if _HAS_TQDM:
        _tqdm.write(msg)
    else:
        print(msg)


# Compatibility names remain public; all new jobs pass TaskContext explicitly.
_terminate_process = terminate_process
FFmpegCancellation = ProcessCancellation


class PositionQueue(Queue):
    """Legacy callback adapter; internal ownership resides in TaskContext."""
    def __init__(self, context):
        super().__init__()
        self.context = context

    @property
    def cancellation(self):
        return self.context.cancellation


def run_ffmpeg_with_progress(cmd, info, desc, position_q, debug_log_path=None,
                             progress_callback=None, *, task_context=None, status_callback=None):
    if task_context is None:
        task_context = getattr(position_q, "context", None)
        if task_context is None and getattr(position_q, "cancellation", None) is not None:
            from auto_reframe_core.job_models import TaskContext
            task_context = TaskContext(cancellation=position_q.cancellation)
    position = position_q.get()
    last_step = [-1]

    def report(percent):
        if progress_callback is not None:
            progress_callback(percent)
        elif percent // 5 > last_step[0]:
            last_step[0] = percent // 5
            print(f"[進度] {desc}：{percent // 5 * 5}%")
    try:
        return ProcessRunner(task_context).ffmpeg(cmd, info, report, status=status_callback,
                                                  debug_path=debug_log_path)
    finally:
        position_q.put(position)


def run_parallel(tasks, process_fn, workers, progress_callback=None, *, task_context=None, detailed=False):
    from auto_reframe_core.job_models import TaskContext, VideoResult, BatchResult, ResultStatus
    from auto_reframe_core.job_errors import describe_error
    context = task_context or TaskContext()
    positions = PositionQueue(context)
    for index in range(workers):
        positions.put(index)
    results = {}
    for task in tasks:
        emit_video_progress(progress_callback, task, kind="queued", progress=0, phase="等待處理")

    def execute_task(task):
        context.check()
        emit_video_progress(progress_callback, task, kind="started", progress=0, phase="讀取影片資訊")
        return process_fn(task, positions)

    def collect(task, future):
        path = Path(task[2])
        message, error_kind = "", ""
        try:
            success = future.result()
        except TaskCancelled:
            success = False
        except Exception as exc:
            success = False
            error_kind, message = describe_error(exc)
        targets = tuple(context.target_results.get(path, {}).values())
        if success:
            status = (ResultStatus.SKIPPED if targets and all(target.status == ResultStatus.SKIPPED for target in targets)
                      else ResultStatus.SUCCESS)
        elif context.cancellation.cancelled:
            status = ResultStatus.CANCELLED
        elif any(target.status == ResultStatus.SUCCESS for target in targets):
            status = ResultStatus.PARTIAL
        else:
            status = ResultStatus.FAILED
        if not success and not message:
            error_kind, message = context.errors.get(path, ("cancelled", "工作已取消") if status == ResultStatus.CANCELLED else ("encoding", "處理未完成"))
        expected = getattr(context, "planned_targets", {}).get(path, ())
        missing = [target for target in expected if target not in {entry.path for entry in targets}]
        context.record_targets(path, missing, status, message, error_kind)
        targets = tuple(context.target_results.get(path, {}).values())
        results[path] = VideoResult(path, status, targets, error_kind, message)
        kind = {ResultStatus.SUCCESS: "completed", ResultStatus.SKIPPED: "skipped",
                ResultStatus.CANCELLED: "cancelled"}.get(status, "failed")
        emit_video_progress(progress_callback, task, kind=kind,
                            progress=100 if success else None, message=message)

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    interrupted = False
    try:
        futures = {executor.submit(execute_task, task): task for task in tasks}
        for future in as_completed(futures):
            collect(futures[future], future)
    except KeyboardInterrupt:
        interrupted = True
        context.cancellation.cancel()
        for future in futures:
            future.cancel()
        if not detailed:
            raise
    finally:
        executor.shutdown(wait=True, cancel_futures=interrupted)
    for future, task in futures.items():
        if Path(task[2]) not in results:
            if future.cancelled():
                results[Path(task[2])] = VideoResult(Path(task[2]), ResultStatus.CANCELLED)
            else:
                collect(task, future)
    batch = BatchResult(context.job_id, tuple(results[Path(task[2])] for task in tasks))
    return batch if detailed else batch.legacy()
