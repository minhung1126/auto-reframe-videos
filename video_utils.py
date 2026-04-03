import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

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

def detect_hw_encoder(ffmpeg_path: str = "ffmpeg") -> Tuple[str, Optional[str]]:
    """偵測可用的硬體加速編碼器"""
    try:
        res = subprocess.run([ffmpeg_path, "-hide_banner", "-encoders"], 
                             capture_output=True, text=True, timeout=10)
        encoders = res.stdout
    except Exception:
        print(f"[錯誤] 呼叫 {ffmpeg_path} 失敗，請確認其是否存在。")
        sys.exit(1)

    # 優先順序: NVENC > AMF > QSV
    for enc, hw in [("h264_nvenc", "cuda"), ("h264_amf", "d3d11va"), ("h264_qsv", "qsv")]:
        if enc in encoders:
            test = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-f", "lavfi", "-i", 
                 "nullsrc=s=256x256:d=1", "-c:v", enc, "-f", "null", "-"],
                capture_output=True, text=True, timeout=30
            )
            if test.returncode == 0:
                print(f"  [核心系統] 已啟用硬體加速編碼器: {enc} ({hw})")
                return enc, hw

    print("  [核心系統] 未發現可用硬體加速，回退至軟體編碼 (libx264)")
    return "libx264", None


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


def get_youtube_bitrate(short_side: int, fps: float) -> str:
    """根據 YouTube 標準建議與影像短邊高度決定 bitrate"""
    high_fps = fps > 30
    if short_side >= 2160: return "60M" if high_fps else "40M"
    if short_side >= 1440: return "24M" if high_fps else "16M"
    if short_side >= 1080: return "12M" if high_fps else "8M"
    if short_side >= 720: return "7500K" if high_fps else "5M"
    if short_side >= 480: return "4M" if high_fps else "2500K"
    return "1500K" if high_fps else "1M"


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
