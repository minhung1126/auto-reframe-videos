import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Tuple, Optional, Dict, Any, List

from auto_reframe_core.encoder_profiles import (
    build_encoder_args,
    build_output_args,
    detect_h264_hw_encoder,
    detect_h265_hw_encoder,
    detect_hwaccel_for_cmd,
    double_bitrate,
)
from auto_reframe_core.platform_profile import resolve_workers

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


def get_video_info(ffprobe_path: str, input_file: Path) -> Optional[Dict[str, Any]]:
    """呼叫 FFprobe 獲取影片解析度、時長、FPS 等資訊"""
    try:
        res = subprocess.run(
            [ffprobe_path, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(input_file)],
            capture_output=True, text=True, timeout=30
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

    width = _safe_int(v_stream.get("width"), 0)
    height = _safe_int(v_stream.get("height"), 0)
    if width <= 0 or height <= 0:
        print(f"  [警告] 影片 {input_file.name} 解析度異常（{width}x{height}），已跳過。")
        return None

    fps_str = v_stream.get("r_frame_rate", "30/1")
    fps = parse_fps(fps_str)

    duration = _safe_float(data.get("format", {}).get("duration", 0), 0.0)

    return {
        "width": width,
        "height": height,
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
    import time
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
# 以下為 auto_compress.py 與 auto_reframe.py 共用的常數與工具函式
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


def run_ffmpeg_with_progress(
    cmd: List[str],
    info: Dict[str, Any],
    desc: str,
    position_q: Queue,
    debug_log_path=None,
) -> Tuple[int, List[str]]:
    """
    以 Popen 執行 FFmpeg 指令，並透過 tqdm（或純文字）顯示即時進度。
    回傳 (returncode, stderr_log_tail)。
    position_q 用於 tqdm 多列進度條的位置管理，執行完畢後自動歸還。
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, encoding="utf-8", errors="replace",
    )

    pos = position_q.get()
    pbar = None
    if _HAS_TQDM:
        pbar = _tqdm(
            total=info["duration"], desc=desc, position=pos, leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {elapsed}<{remaining}",
        )
    else:
        sys.stdout.write(f"\n{desc} 處理中...")
        sys.stdout.flush()

    stderr_log: List[str] = []
    debug_fd = None
    try:
        if debug_log_path:
            debug_fd = open(debug_log_path, "w", encoding="utf-8")
            debug_fd.write(f"{' '.join(shlex.quote(s) for s in cmd)}\n\n")

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

        proc.wait()
    except Exception:
        proc.terminate()
        proc.wait()
        raise
    finally:
        if pbar:
            pbar.close()
        if debug_fd:
            debug_fd.close()
        position_q.put(pos)

    return proc.returncode, stderr_log


def run_parallel(
    tasks: list,
    process_fn,
    workers: int,
) -> Tuple[int, List[str]]:
    """
    以 ThreadPoolExecutor 平行執行任務並彙整結果。
    process_fn 簽名：(task_info, position_q) -> bool
    task_info 格式：(idx, total, Path)
    回傳 (success_count, failed_file_names)。
    """
    position_q: Queue = Queue()
    for i in range(workers):
        position_q.put(i)

    success_count = 0
    failed_files: List[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_fn, t, position_q): t for t in tasks}
        try:
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    if fut.result():
                        success_count += 1
                    else:
                        failed_files.append(t[2].name)
                except Exception as e:
                    failed_files.append(t[2].name)
                    print(f"\n[錯誤] 處理 {t[2].name} 時發生異常: {e}")
        except KeyboardInterrupt:
            for fut in futures:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            print("\n[中斷] 已收到停止指令，正在停止尚未開始的任務。")
            raise

    return success_count, failed_files
