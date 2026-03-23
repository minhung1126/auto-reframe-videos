# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
支援硬體加速、文字疊加、多比例輸出、批次處理
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# 強制 stdout/stderr 使用 UTF-8，避免 Windows cp950 編碼問題
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# ==================== 使用者設定區 ============================
# ============================================================

# --- 輸入 / 輸出 ---
INPUT_DIR = "input"        # 輸入資料夾路徑，會處理此資料夾內所有影片
OUTPUT_DIR = "output"      # 輸出資料夾路徑

# --- 裁切比例 (寬:高) ---
# 可同時指定多組，每組會產生一個獨立的輸出檔案
# 例如 [(4, 5), (1, 1)] 會分別產生 4:5 和 1:1 兩種裁切
TARGET_RATIOS = [(4, 5), (1, 1)]

# --- 最終輸出比例 (寬:高) ---
# 裁切後會上下補黑邊至此比例
FINAL_RATIO = (9, 16)

# --- 上方文字設定 ---
# 從文字檔讀取，顯示在裁切區域正上方的黑邊區域
# 支援多行，檔案內直接換行即可
# 檔案為空或不存在則不顯示
TOP_TEXT_FILE = "top_text.txt"   # 上方文字檔案路徑（相對於腳本所在目錄）
TOP_FONT_SIZE = 48               # 上方文字大小 (px)

# --- 下方文字設定 ---
# 從文字檔讀取，顯示在裁切區域正下方的黑邊區域
# 支援多行，檔案內直接換行即可
# 檔案為空或不存在則不顯示
BOTTOM_TEXT_FILE = "bottom_text.txt"  # 下方文字檔案路徑（相對於腳本所在目錄）
BOTTOM_FONT_SIZE = 24                  # 下方文字大小 (px)

# --- 字型設定 ---
FONT_PATH = "fonts/NotoSerifTC.ttf"   # 字型檔案路徑（相對於腳本所在目錄）
FONT_COLOR = "white"                   # 字型顏色

# --- 文字邊距 ---
TEXT_MARGIN = 20           # 文字與裁切區域邊緣的間距 (px)
TEXT_LINE_SPACING = 1.2    # 多行文字的行距倍率 (基於字體高度)

# --- ffmpeg / ffprobe 路徑 ---
FFMPEG_PATH = "ffmpeg"     # ffmpeg 執行檔路徑（若已加入 PATH 則直接寫 "ffmpeg"）
FFPROBE_PATH = "ffprobe"   # ffprobe 執行檔路徑

# --- 支援的影片副檔名 ---
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"}

# ============================================================
# ==================== 以下為程式邏輯 ==========================
# ============================================================


def detect_hw_encoder():
    """
    偵測系統支援的硬體加速編碼器。
    優先順序：NVENC > AMF > QSV > 軟體 (libx264)
    回傳 (encoder, hwaccel_type_or_None)
    """
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        encoders_output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[錯誤] 找不到 ffmpeg，請確認 FFMPEG_PATH 設定正確。")
        sys.exit(1)

    # 優先順序：NVENC → AMF → QSV → libx264
    hw_options = [
        ("h264_nvenc", "cuda"),
        ("h264_amf",   "d3d11va"),
        ("h264_qsv",   "qsv"),
    ]

    for encoder, hwaccel in hw_options:
        if encoder in encoders_output:
            # 進一步驗證編碼器是否真的可用（有些系統列出但實際不可用）
            test = subprocess.run(
                [FFMPEG_PATH, "-hide_banner", "-f", "lavfi", "-i",
                 "nullsrc=s=256x256:d=1", "-c:v", encoder,
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=15
            )
            if test.returncode == 0:
                print(f"[硬體加速] 偵測到: {encoder} (hwaccel={hwaccel})")
                return encoder, hwaccel

    print("[硬體加速] 未偵測到硬體加速器，使用軟體編碼 (libx264)")
    return "libx264", None


def get_video_info(input_file):
    """
    使用 ffprobe 讀取影片資訊。
    回傳 dict: {width, height, fps, duration, has_audio, audio_codec}
    """
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(input_file)],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"[錯誤] 無法使用 ffprobe 讀取: {input_file}")
        return None
    except json.JSONDecodeError:
        print(f"[錯誤] ffprobe 輸出解析失敗: {input_file}")
        return None

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        print(f"[錯誤] 找不到視訊串流: {input_file}")
        return None

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # 解析 fps
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    # 解析時長
    duration = float(data.get("format", {}).get("duration", 0))

    info = {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": duration,
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }
    return info


def calculate_dimensions(src_w, src_h, target_ratio, final_ratio):
    """
    計算裁切與補黑邊的所有尺寸。

    第一階段：以原影片短邊為基準，按 target_ratio 裁切
    第二階段：以裁切結果的短邊為基準，按 final_ratio 補黑邊

    回傳 dict:
        crop_w, crop_h       — 第一階段裁切尺寸
        crop_x, crop_y       — 裁切起始座標（正中央裁切）
        pad_top, pad_bottom   — 上下補黑邊量
        final_w, final_h     — 最終輸出尺寸（補黑邊後）
    """
    t_w, t_h = target_ratio

    # 第一階段：以短邊為新的長邊，計算新短邊
    short_side = min(src_w, src_h)  # 原影片短邊

    # 新長邊 = 短邊，新短邊 = 短邊 * t_w / t_h（確保比例 t_w:t_h，且 t_w < t_h 時短邊在上）
    # 因為輸出是直式，crop_w < crop_h
    if t_w <= t_h:
        crop_h = short_side
        crop_w = int(short_side * t_w / t_h)
    else:
        crop_w = short_side
        crop_h = int(short_side * t_h / t_w)

    # 確保裁切不超過原影片尺寸
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    # 確保尺寸為偶數（ffmpeg 要求）
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    # 裁切起始座標（正中央）
    crop_x = (src_w - crop_w) // 2
    crop_y = (src_h - crop_h) // 2

    # 第二階段：補黑邊至 final_ratio (9:16)
    f_w, f_h = final_ratio

    # 以裁切後短邊（寬度）為基準
    final_w = crop_w
    final_h = int(crop_w * f_h / f_w)

    # 確保最終高度為偶數
    final_h = final_h + (final_h % 2)

    # 上下各補的黑邊量
    total_pad = final_h - crop_h
    pad_top = total_pad // 2
    pad_bottom = total_pad - pad_top

    # 確保 pad 值非負
    pad_top = max(0, pad_top)
    pad_bottom = max(0, pad_bottom)

    return {
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "final_w": final_w,
        "final_h": final_h,
    }


def select_resolution(final_w, final_h):
    """
    根據最終尺寸選擇最適合的標準解析度。
    直式影片以短邊（寬度）為基準：
      4K: 短邊 >= 2160 → 2160×3840
      2K: 短邊 >= 1440 → 1440×2560
      FHD: 保底 → 1080×1920
    優先選大的，至少 FHD。
    回傳 (out_w, out_h, label)
    """
    short = min(final_w, final_h)

    if short >= 2160:
        return 2160, 3840, "4K"
    elif short >= 1440:
        return 1440, 2560, "2K"
    else:
        return 1080, 1920, "FHD"


def select_bitrate(out_h, fps):
    """
    根據解析度長邊與幀率選擇 YouTube 推薦的位元率。
    回傳 video_bitrate 字串（例如 "8M"）
    """
    high_fps = fps > 30

    if out_h >= 3840:
        return "60M" if high_fps else "40M"
    elif out_h >= 2560:
        return "24M" if high_fps else "16M"
    else:
        return "12M" if high_fps else "8M"


def _load_text_from_file(filepath):
    """
    從文字檔讀取內容，回傳 (清理後的字串, 是否為新建檔案)。
    - 使用 UTF-8 編碼讀取
    - 移除末尾多餘的空白行
    - 檔案不存在則自動建立空檔，回傳 ("", True)
    """
    script_dir = Path(__file__).resolve().parent
    text_path = script_dir / filepath

    if not text_path.exists():
        text_path.write_text("", encoding="utf-8")
        print(f"  [提示] 文字檔不存在，已自動建立: {text_path}")
        return "", True

    try:
        text = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 回退嘗試 UTF-8 with BOM
        try:
            text = text_path.read_text(encoding="utf-8-sig")
        except Exception:
            print(f"  [警告] 無法讀取文字檔: {text_path}")
            return "", False

    # 移除末尾空白行
    text = text.rstrip("\r\n")
    return (text if text.strip() else ""), False


def build_ffmpeg_command(
    input_file, output_file,
    video_info, dims,
    out_w, out_h, video_bitrate,
    encoder, hwaccel,
):
    """
    構建 ffmpeg 指令。
    完成：裁切 → 補黑邊 → 縮放 → 文字疊加 → 編碼
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-y"]

    # 硬體加速解碼
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]

    cmd += ["-i", str(input_file)]

    # ---- 構建 filter_complex ----
    filters = []

    # 1. 裁切（正中央）
    filters.append(
        f"crop={dims['crop_w']}:{dims['crop_h']}:{dims['crop_x']}:{dims['crop_y']}"
    )

    # 2. 補黑邊
    if dims["pad_top"] > 0 or dims["pad_bottom"] > 0:
        pad_w = dims["final_w"]
        pad_h = dims["final_h"]
        pad_x = 0
        pad_y = dims["pad_top"]
        filters.append(f"pad={pad_w}:{pad_h}:{pad_x}:{pad_y}:black")

    # 3. 縮放到目標解析度
    filters.append(f"scale={out_w}:{out_h}:flags=lanczos")

    # 4. 文字疊加
    # 解析字型路徑（ffmpeg drawtext 在 Windows 上需要特殊處理路徑）
    script_dir = Path(__file__).resolve().parent
    font_abs = script_dir / FONT_PATH
    # ffmpeg drawtext 需要用 / 並且冒號要轉義
    font_path_ffmpeg = str(font_abs).replace("\\", "/").replace(":", "\\:")

    top_text = _load_text_from_file(TOP_TEXT_FILE)[0]
    bottom_text = _load_text_from_file(BOTTOM_TEXT_FILE)[0]

    # 以 1920 作為高度基準計算縮放比例，確保字體大小和邊界在不同輸出解析度下視覺一致
    font_scale = out_h / 1920.0
    scaled_top_font = int(TOP_FONT_SIZE * font_scale)
    scaled_bottom_font = int(BOTTOM_FONT_SIZE * font_scale)
    scaled_margin = int(TEXT_MARGIN * font_scale)
    
    # 計算外框粗細以達到加粗效果
    scaled_border_w_top = max(1, int(scaled_top_font * 0.03))
    scaled_border_w_bottom = max(1, int(scaled_bottom_font * 0.03))

    if top_text:
        top_lines = top_text.splitlines()
        if top_lines:
            # 計算裁切區域在補黑邊後的頂部 y 座標（縮放後）
            scale_factor = out_h / dims["final_h"]
            pad_top_scaled = int(dims["pad_top"] * scale_factor)
            
            n_lines = len(top_lines)
            for i, line in enumerate(top_lines):
                escaped = line.replace("'", "'\\''").replace(":", "\\:")
                # 上方文字：在黑邊區域內，向下靠齊（靠近裁切區域邊緣）
                y_expr = f"{pad_top_scaled}-{scaled_margin}-text_h-({n_lines - 1 - i})*line_h*{TEXT_LINE_SPACING}"
                
                filters.append(
                    f"drawtext=fontfile='{font_path_ffmpeg}'"
                    f":text='{escaped}'"
                    f":fontsize={scaled_top_font}"
                    f":fontcolor={FONT_COLOR}"
                    f":borderw={scaled_border_w_top}"
                    f":bordercolor={FONT_COLOR}"
                    f":x=(w-text_w)/2"
                    f":y={y_expr}"
                )

    if bottom_text:
        bottom_lines = bottom_text.splitlines()
        if bottom_lines:
            scale_factor = out_h / dims["final_h"]
            pad_bottom_scaled = int(dims["pad_bottom"] * scale_factor)

            n_lines = len(bottom_lines)
            for i, line in enumerate(bottom_lines):
                # 下方文字：在黑邊區域內，向上靠齊（靠近裁切區域邊緣）
                escaped = line.replace("'", "'\\''").replace(":", "\\:")
                y_expr = f"{out_h}-{pad_bottom_scaled}+{scaled_margin}+{i}*line_h*{TEXT_LINE_SPACING}"
                
                filters.append(
                    f"drawtext=fontfile='{font_path_ffmpeg}'"
                    f":text='{escaped}'"
                    f":fontsize={scaled_bottom_font}"
                    f":fontcolor={FONT_COLOR}"
                    f":borderw={scaled_border_w_bottom}"
                    f":bordercolor={FONT_COLOR}"
                    f":x=(w-text_w)/2"
                    f":y={y_expr}"
                )

    vf = ",".join(filters)
    cmd += ["-vf", vf]

    # ---- 視訊編碼 ----
    cmd += ["-c:v", encoder]
    cmd += ["-b:v", video_bitrate]

    # 編碼器特定參數
    if encoder == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    elif encoder == "h264_amf":
        cmd += ["-quality", "balanced", "-rc", "vbr_latency"]
    elif encoder == "h264_qsv":
        cmd += ["-preset", "medium", "-global_quality", "22"]
    else:
        # libx264
        cmd += ["-preset", "medium", "-crf", "18", "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate.rstrip("M")) * 2) + "M"]

    cmd += ["-pix_fmt", "yuv420p"]

    # ---- 音訊 (AAC 192kbps) ----
    if video_info["has_audio"]:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    # ---- 輸出 ----
    cmd += ["-movflags", "+faststart"]  # MP4 優化，讓影片可以邊下載邊播放
    cmd += [str(output_file)]

    return cmd





def process_single_video(input_file, encoder, hwaccel):
    """處理單一影片檔案"""
    print(f"\n{'='*60}")
    print(f"[處理] {input_file.name}")
    print(f"{'='*60}")

    # 讀取影片資訊
    info = get_video_info(input_file)
    if not info:
        return False

    print(f"  原始尺寸: {info['width']}×{info['height']}")
    print(f"  幀率: {info['fps']} fps")
    print(f"  時長: {info['duration']:.1f} 秒")
    print(f"  音訊: {'有' if info['has_audio'] else '無'} ({info['audio_codec'] or 'N/A'})")

    stem = input_file.stem
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for t_w, t_h in TARGET_RATIOS:
        print(f"\n  --- 目標比例: {t_w}:{t_h} ---")

        # 計算尺寸
        dims = calculate_dimensions(info["width"], info["height"], (t_w, t_h), FINAL_RATIO)
        print(f"  裁切: {dims['crop_w']}×{dims['crop_h']}"
              f" (起點: {dims['crop_x']},{dims['crop_y']})")
        print(f"  補黑邊: 上 {dims['pad_top']}px / 下 {dims['pad_bottom']}px")
        print(f"  補黑邊後: {dims['final_w']}×{dims['final_h']}")

        # 選擇解析度
        out_w, out_h, res_label = select_resolution(dims["final_w"], dims["final_h"])
        print(f"  輸出解析度: {out_w}×{out_h} ({res_label})")

        # 選擇位元率
        vbr = select_bitrate(out_h, info["fps"])
        print(f"  視訊位元率: {vbr}")

        # 輸出檔名：原檔名_比例.mp4
        output_file = output_dir / f"{stem}_{t_w}x{t_h}.mp4"

        # 構建指令
        cmd = build_ffmpeg_command(
            input_file, output_file,
            info, dims,
            out_w, out_h, vbr,
            encoder, hwaccel,
        )

        print(f"  輸出: {output_file}")

        # 執行 ffmpeg
        print("  [執行中] 編碼中...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [錯誤] 編碼失敗:\n{result.stderr[-500:]}")
            ok = False
        else:
            print("  [成功] 編碼完成")
            ok = True

        if ok:
            success_count += 1

    return success_count == len(TARGET_RATIOS)


def main():
    print("=" * 60)
    print("  Auto Reframe Video — 橫轉直影片工具")
    print("=" * 60)

    # 檢查輸入資料夾
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        print(f"\n[提示] 輸入資料夾 '{INPUT_DIR}' 不存在，已自動建立。")
        print(f"       請將要轉換的影片放入 '{input_dir.resolve()}' 後重新執行。")
        sys.exit(0)

    # 掃描影片檔案
    video_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"\n[提示] 輸入資料夾 '{input_dir.resolve()}' 內沒有找到影片檔案。")
        print(f"       支援的格式: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(0)

    print(f"\n找到 {len(video_files)} 個影片檔案:")
    for f in video_files:
        print(f"  • {f.name}")

    # 顯示設定摘要
    print(f"\n[設定]")
    print(f"  目標裁切比例: {', '.join(f'{w}:{h}' for w, h in TARGET_RATIOS)}")
    print(f"  最終比例: {FINAL_RATIO[0]}:{FINAL_RATIO[1]}")
    top_text, top_created = _load_text_from_file(TOP_TEXT_FILE)
    bottom_text, bottom_created = _load_text_from_file(BOTTOM_TEXT_FILE)
    if top_text:
        print(f"  上方文字: {repr(top_text)}")
    else:
        print(f"  上方文字: (無，檔案: {TOP_TEXT_FILE})")
    if bottom_text:
        print(f"  下方文字: {repr(bottom_text)}")
    else:
        print(f"  下方文字: (無，檔案: {BOTTOM_TEXT_FILE})")

    # 若有新建文字檔，暫停讓使用者編輯
    if top_created or bottom_created:
        print(f"\n[提示] 已自動建立文字檔，請先編輯後再繼續。")
        input("按任意鍵繼續...")

    # 偵測硬體加速
    print()
    encoder, hwaccel = detect_hw_encoder()

    # 處理每一個影片
    total = len(video_files)
    success = 0
    failed = []

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{total}] 處理中...")
        if process_single_video(video_file, encoder, hwaccel):
            success += 1
        else:
            failed.append(video_file.name)

    # 結果摘要
    print(f"\n{'='*60}")
    print(f"  處理完成！")
    print(f"  成功: {success}/{total}")
    if failed:
        print(f"  失敗: {', '.join(failed)}")
    print(f"  輸出目錄: {Path(OUTPUT_DIR).resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
