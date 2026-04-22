import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Tuple, Optional, Dict, Any, List

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

# 各 codec 的硬體加速編碼器候選清單（優先順序：NVENC > AMF > QSV > VideoToolbox）
# VideoToolbox 為 macOS 原生 GPU 加速（支援 Apple Silicon / Intel Mac）
_HW_ENCODER_CANDIDATES = {
    "h265": [
        ("hevc_nvenc",        "cuda"),          # NVIDIA (Windows/Linux)
        ("hevc_amf",         "d3d11va"),        # AMD (Windows)
        ("hevc_qsv",         "qsv"),            # Intel Quick Sync (Windows/Linux)
        ("hevc_videotoolbox", "videotoolbox"),  # Apple VideoToolbox (macOS)
    ],
    "h264": [
        ("h264_nvenc",        "cuda"),          # NVIDIA (Windows/Linux)
        ("h264_amf",         "d3d11va"),        # AMD (Windows)
        ("h264_qsv",         "qsv"),            # Intel Quick Sync (Windows/Linux)
        ("h264_videotoolbox", "videotoolbox"),  # Apple VideoToolbox (macOS)
    ],
}
_SW_FALLBACK = {"h265": "libx265", "h264": "libx264"}


def _detect_hw_encoder(ffmpeg_path: str, codec: str) -> Tuple[str, Optional[str]]:
    """通用硬體加速編碼器偵測，支援 'h265' 或 'h264'。"""
    try:
        res = subprocess.run([ffmpeg_path, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10)
        encoders = res.stdout
    except Exception:
        print(f"[錯誤] 呼叫 {ffmpeg_path} 失敗，請確認其是否存在。")
        sys.exit(1)

    label = codec.upper()  # 用於顯示訊息
    for enc, hw in _HW_ENCODER_CANDIDATES[codec]:
        if enc in encoders:
            test = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-f", "lavfi", "-i",
                 "nullsrc=s=256x256:d=1", "-c:v", enc, "-f", "null", "-"],
                capture_output=True, text=True, timeout=30
            )
            if test.returncode == 0:
                print(f"  [核心系統] 已啟用硬體加速編碼器 ({label}): {enc} ({hw})")
                return enc, hw

    sw = _SW_FALLBACK[codec]
    print(f"  [核心系統] 未發現可用硬體加速 ({label})，回退至軟體編碼 ({sw})")
    return sw, None


def detect_h265_hw_encoder(ffmpeg_path: str = "ffmpeg") -> Tuple[str, Optional[str]]:
    """偵測可用的 H.265 硬體加速編碼器"""
    return _detect_hw_encoder(ffmpeg_path, "h265")


def detect_h264_hw_encoder(ffmpeg_path: str = "ffmpeg") -> Tuple[str, Optional[str]]:
    """偵測可用的 H.264 硬體加速編碼器"""
    return _detect_hw_encoder(ffmpeg_path, "h264")


def get_video_info(ffprobe_path: str, input_file: Path) -> Optional[Dict[str, Any]]:
    """呼叫 FFprobe 獲取影片解析度、時長、FPS 等資訊"""
    try:
        res = subprocess.run(
            [ffprobe_path, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(input_file)],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(res.stdout)
    except Exception as e:
        print(f"  [警告] 無法取得影片資訊: {input_file.name} ({e})")
        return None

    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    if not v_stream:
        print(f"  [警告] 影片 {input_file.name} 中未找到視訊串流，已跳過。")
        return None

    fps_str = v_stream.get("r_frame_rate", "30/1")
    fps = parse_fps(fps_str)
    
    return {
        "width": int(v_stream.get("width", 0)),
        "height": int(v_stream.get("height", 0)),
        "fps": round(fps, 3),
        "duration": float(data.get("format", {}).get("duration", 0)),
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


def double_bitrate(vbr: str) -> str:
    """安全地將 bitrate 數值倍增（如 '12M' → '24M'，相容 M/K 或小數）"""
    m = re.fullmatch(r"([\d\.]+)([A-Za-z]+)", str(vbr))
    if not m:
        raise ValueError(f"無法解析 bitrate 字串: {vbr!r}")
    val = float(m.group(1)) * 2
    # 移除小數點後為 0 的 .0
    val_str = f"{int(val)}" if val.is_integer() else f"{val}"
    return f"{val_str}{m.group(2)}"


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
        print(f"  [清理] 發現 {len(tmp_files)} 個殘留暫存檔，正在刪除...")
        for tmp in tmp_files:
            try:
                tmp.unlink()
            except OSError:
                pass


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
    """給定解析度關鍵字與來源短邊，回傳最終輸出短邊（不放大、向下取最接近的標準值）。"""
    target_limit = RESOLUTION_MAP.get(res_key.lower(), 1080)
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


def resolve_workers(max_workers: int) -> int:
    """將 max_workers 設定解析為實際使用的執行緒數（自動偵測，macOS 上限為 8，其餘為 4）。"""
    w = max_workers
    if w <= 0:
        w = (os.cpu_count() or 2) // 2
    
    # M-series Mac 媒體引擎強大，可允許更多平行任務
    limit = 8 if sys.platform == "darwin" else 4
    return max(1, min(w, limit))


def detect_hwaccel_for_cmd(hwaccels: set) -> List[str]:
    """
    給定一組 hwaccel 後端字串，回傳應插入 FFmpeg 命令的解碼端 hwaccel 旗標。
    """
    if len(hwaccels) != 1:
        return []
    hw = next(iter(hwaccels))
    # 支援 VideoToolbox, CUDA, QSV 等硬體解碼
    return ["-hwaccel", hw]


def build_encoder_args(encoder: str, vbr: str) -> List[str]:
    """根據編碼器類型回傳對應的 FFmpeg 視訊編碼參數清單（不含輸出檔路徑）。"""
    v = "v:0"
    if encoder in ("hevc_nvenc", "h264_nvenc"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-preset", "p4", "-rc", "vbr"]
    if encoder in ("hevc_amf", "h264_amf"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-quality", "balanced", "-rc", "vbr_latency"]
    if encoder in ("hevc_qsv", "h264_qsv"):
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-preset", "medium"]
    if encoder in ("hevc_videotoolbox", "h264_videotoolbox"):
        # VideoToolbox 參數優化
        return [f"-c:{v}", encoder, f"-b:{v}", vbr, "-allow_sw", "1"]
    # libx265 / libx264 軟體編碼
    return [f"-c:{v}", encoder, "-preset", "medium", f"-b:{v}", vbr,
            f"-maxrate:{v}", double_bitrate(vbr), f"-bufsize:{v}", double_bitrate(vbr)]


def build_output_args(encoder: str, vbr: str, vcodec: str,
                      has_audio: bool, out_file) -> List[str]:
    """
    組合完整的單路輸出參數：
    視訊編碼 + pix_fmt + hvc1 tag（h265）+ 音訊（aac 192k）+ mp4 容器。
    不含 -map 旗標，呼叫端需自行在前面加入。
    """
    args = build_encoder_args(encoder, vbr)
    args += ["-pix_fmt", "yuv420p"]
    if vcodec == h265:
        args += ["-tag:v", "hvc1"]
    if has_audio:
        args += ["-c:a:0", "aac", "-b:a:0", "192k"]
    args += ["-f", "mp4", "-movflags", "+faststart", str(out_file)]
    return args


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
                m = re.search(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", line)
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

    return success_count, failed_files
