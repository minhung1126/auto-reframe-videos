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


def get_video_info(ffprobe_path: str, input_file: Path) -> Optional[Dict[str, Any]]:
    """呼叫 FFprobe 獲取影片資訊；width/height 為 FFmpeg autorotate 後的顯示尺寸。"""
    try:
        res = subprocess.run(
            [ffprobe_path, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(input_file)],
            capture_output=True, text=True, timeout=30,
            **hidden_subprocess_kwargs(),
        )
        if res.returncode != 0 or not res.stdout.strip():
            print(f"  [警告] FFprobe 無法解析影片: {input_file.name}")
            return None
        data = json.loads(res.stdout)
    except Exception as e:
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


def _terminate_process(process: subprocess.Popen, timeout: float = 3.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait()
        except OSError:
            pass


class FFmpegCancellation:
    """Thread-safe cancellation state and registry for active FFmpeg processes."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._processes = set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def active_process_count(self) -> int:
        with self._lock:
            return len(self._processes)

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def register(self, process: subprocess.Popen) -> None:
        terminate_immediately = False
        with self._lock:
            if self._event.is_set():
                terminate_immediately = True
            else:
                self._processes.add(process)
        if terminate_immediately:
            _terminate_process(process)

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel(self, timeout: float = 3.0) -> None:
        """Stop every registered child, escalating to kill after one shared deadline."""
        self._event.set()
        with self._lock:
            processes = list(self._processes)

        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + max(0.0, timeout)
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                break

        for process in processes:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass


def run_ffmpeg_with_progress(
    cmd: List[str],
    info: Dict[str, Any],
    desc: str,
    position_q: Queue,
    debug_log_path=None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[int, List[str]]:
    """
    以 Popen 執行 FFmpeg 指令，並透過 callback、tqdm 或純文字顯示即時進度。
    回傳 (returncode, stderr_log_tail)。
    position_q 用於 tqdm 多列進度條的位置管理，執行完畢後自動歸還。
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, encoding="utf-8", errors="replace",
        **hidden_subprocess_kwargs(),
    )
    cancellation = getattr(position_q, "cancellation", None)
    if cancellation is not None:
        cancellation.register(proc)

    pos = None
    pbar = None
    stderr_log: List[str] = []
    debug_fd = None
    try:
        pos = position_q.get()
        use_tqdm = progress_callback is None and _HAS_TQDM and sys.stderr.isatty()
        progress_step = -1
        if progress_callback is not None:
            progress_callback(0)
            progress_step = 0
        elif use_tqdm:
            pbar = _tqdm(
                total=info["duration"], desc=desc, position=pos, leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {elapsed}<{remaining}",
            )
        else:
            print(f"[處理] {desc}：0%")

        if debug_log_path:
            debug_fd = open(debug_log_path, "w", encoding="utf-8")
            debug_fd.write(f"{' '.join(shlex.quote(s) for s in cmd)}\n\n")

        if proc.stdout is not None:
            for line in proc.stdout:
                stderr_log.append(line)
                if len(stderr_log) > 15:
                    stderr_log.pop(0)
                if debug_fd:
                    debug_fd.write(line)
                    debug_fd.flush()
                if "time=" in line:
                    # 支援變長的小時數與 1~3 位的毫秒數 (例如 00:00:05.3 或 00:00:05.123)
                    m = re.search(r"time=(\d+:\d{2}:\d{2}(?:\.\d+)?)", line)
                    if m and pbar:
                        pbar.n = min(parse_ffmpeg_time(m.group(1)), info["duration"])
                        pbar.refresh()
                    elif m:
                        elapsed = min(parse_ffmpeg_time(m.group(1)), info["duration"])
                        duration = info["duration"]
                        percent = int(elapsed * 100 / duration) if duration else 0
                        step = min(100, percent // 5 * 5)
                        if step > progress_step:
                            progress_step = step
                            if progress_callback is not None:
                                progress_callback(step)
                            else:
                                print(f"[進度] {desc}：{step}%")

        proc.wait()
        if proc.returncode == 0 and progress_callback is not None and progress_step < 100:
            progress_callback(100)
    except BaseException:
        _terminate_process(proc)
        raise
    finally:
        if pbar:
            pbar.close()
        if debug_fd:
            debug_fd.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if pos is not None:
            position_q.put(pos)
        if cancellation is not None:
            cancellation.unregister(proc)

    return proc.returncode, stderr_log


def run_parallel(
    tasks: list,
    process_fn,
    workers: int,
    progress_callback: VideoProgressCallback = None,
) -> Tuple[int, List[str]]:
    """
    以 ThreadPoolExecutor 平行執行任務並彙整結果。
    process_fn 簽名：(task_info, position_q) -> bool
    task_info 格式：(idx, total, Path)
    回傳 (success_count, failed_file_names)。
    """
    cancellation = FFmpegCancellation()
    position_q: Queue = Queue()
    position_q.cancellation = cancellation
    for i in range(workers):
        position_q.put(i)
    for task in tasks:
        emit_video_progress(
            progress_callback,
            task,
            kind="queued",
            progress=0,
            phase="等待處理",
        )

    success_count = 0
    failed_files: List[str] = []

    def execute_task(task):
        emit_video_progress(
            progress_callback,
            task,
            kind="started",
            progress=0,
            phase="讀取影片資訊",
        )
        return process_fn(task, position_q)

    executor = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    futures = {}
    try:
        futures = {executor.submit(execute_task, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                if fut.result():
                    success_count += 1
                    emit_video_progress(
                        progress_callback,
                        t,
                        kind="completed",
                        progress=100,
                    )
                else:
                    failed_files.append(t[2].name)
                    emit_video_progress(
                        progress_callback,
                        t,
                        kind="failed",
                        message="處理未完成",
                    )
            except Exception as e:
                failed_files.append(t[2].name)
                emit_video_progress(
                    progress_callback,
                    t,
                    kind="failed",
                    message=str(e),
                )
                print(f"\n[錯誤] 處理 {t[2].name} 時發生異常: {e}")
    except KeyboardInterrupt:
        interrupted = True
        cancellation.cancel()
        for fut in futures:
            fut.cancel()
        print("\n[中斷] 已收到停止指令，正在終止執行中與尚未開始的任務。")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=interrupted)

    return success_count, failed_files
