# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0 - H.265)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：H.265 (HEVC) 高效壓縮、平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from collections import OrderedDict
from video_utils import (
    detect_h265_hw_encoder, detect_h264_hw_encoder, get_video_info, double_bitrate,
    parse_ffmpeg_time, cleanup_tmp_files, get_youtube_bitrate
)

# 嘗試匯入 tqdm 顯示進度條
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("\n[系統提示] 未找到 tqdm 模組，將使用傳統文字輸出進度。 (可執行 pip install tqdm 安裝)")

# 強制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- 全域常數 ---
# 方便在設定中使用，不用加引號
h264 = "h264"
h265 = "h265"

# 解析度關鍵字對應表 (指定短邊長度)
# 例如 1080p 代表短邊為 1080，在 9:16 下高度即為 1920
RESOLUTION_MAP = {
    "4k": 2160,
    "2k": 1440,
    "1080p": 1080,
    "fhd": 1080,
    "720p": 720,
    "hd": 720,
    "480p": 480,
    "360p": 360,
    "source": 99999  # 代表不縮放，直接取原始裁切後短邊
}


@dataclass
class ReframeConfig:
    # --- 輸入 / 輸出 ---
    # 輸入影片的資料夾路徑
    input_dir: str = "input"
    # 輸出影片的資料夾路徑
    output_dir: str = "output"

    # --- 裁切與輸出目標 (重要設定) ---
    # 指定要輸出的比例、解析度、編碼器列表
    # resolution: '4k', '2k', '1080p', 'fhd', '720p', 'hd', '480p', '360p', 'source'
    #           (系統會自動選擇「不大於此解析度」的最大解析度，避免強行放大)
    # vcodec: h264, h265
    #
    # 範例：
    # targets = [
    #     {'ratio': (4, 5), 'resolution': '2k', 'vcodec': h265},
    #     {'ratio': (4, 5), 'resolution': '1080p', 'vcodec': h264},
    #     {'ratio': (1, 1), 'resolution': '1080p', 'vcodec': h265},
    # ]
    targets: List[dict] = field(default_factory=lambda: [
        {'ratio': (4, 5), 'resolution': 'source', 'vcodec': h265},
        {'ratio': (4, 5), 'resolution': '1080p', 'vcodec': h264},
    ])

    # 最終輸出的影片畫布比例 (預設為 9:16 直式)
    final_ratio: Tuple[int, int] = (9, 16)

    # --- 字幕與文字疊加 ---
    # 上方文字內容的來源檔案
    top_text_file: str = "top_text.txt"
    # 上方文字的字型大小
    top_font_size: int = 48

    # 下方文字內容的來源檔案
    bottom_text_file: str = "bottom_text.txt"
    # 下方文字的字型大小
    bottom_font_size: int = 24

    # --- 字型與樣式 ---
    # 字型檔案路徑
    font_path: str = "fonts/NotoSerifTC.ttf"
    # 文字顏色
    font_color: str = "white"
    # 文字與邊框的邊距
    text_margin: int = 20
    # 多行文字間的行距
    text_line_spacing: float = 1.2

    # --- 系統與平行化設定 ---
    # FFmpeg 執行檔路徑
    ffmpeg_path: str = "ffmpeg"
    # FFprobe 執行檔路徑
    ffprobe_path: str = "ffprobe"
    # 支援的影片副檔名集合
    video_extensions: set = field(default_factory=lambda: {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"})
    # 是否跳過已存在的輸出檔案
    skip_existing: bool = True
    # 平行處理的任務數量，設為 0 將自動判斷 (上限為 4)
    max_workers: int = 1
    # 是否開啟除錯模式，開啟時會記錄 FFmpeg 詳細輸出
    debug: bool = False

    # --- 執行階段產生的內部變數 ---
    # 執行階段產生的內容，不需手動設定 (上方文字內容)
    top_text_content: str = ""
    # 執行階段產生的內容，不需手動設定 (下方文字內容)
    bottom_text_content: str = ""


class VideoReframer:
    def __init__(self, config: ReframeConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent

        # 1. 初始化讀取：只讀一次，省去每次迴圈讀檔的 I/O 損耗
        self.load_texts()

        # 2. 偵測可用的硬體加速
        self.h265_encoder, self.h265_hwaccel = detect_h265_hw_encoder(self.config.ffmpeg_path)
        self.h264_encoder, self.h264_hwaccel = detect_h264_hw_encoder(self.config.ffmpeg_path)

    def load_texts(self):
        """讀取上下方的文字檔內容"""
        self.config.top_text_content, top_new = self._load_text_from_file(self.config.top_text_file)
        self.config.bottom_text_content, bottom_new = self._load_text_from_file(self.config.bottom_text_file)
        
        if top_new or bottom_new:
            print(f"\n[提示] 系統已新建 '{self.config.top_text_file}' 或 '{self.config.bottom_text_file}'。")
            print("       這個檔案是用來顯示疊加在輸出的上下黑邊文字，留空則不顯示。")

    def _load_text_from_file(self, filepath: str) -> Tuple[str, bool]:
        text_path = self.script_dir / filepath
        if not text_path.exists():
            text_path.write_text("", encoding="utf-8")
            return "", True
        try:
            text = text_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = text_path.read_text(encoding="utf-8-sig")
            except Exception:
                print(f"  [警告] 無法讀取文字檔: {text_path}")
                return "", False
        return text.rstrip("\r\n"), False



    def calculate_dimensions(self, src_w, src_h, target_ratio):
        t_w, t_h = target_ratio
        
        # 使用比例判定裁切基準，確保橫式與直式影片都能正確處理
        source_ratio = src_w / src_h
        target_ratio_val = t_w / t_h
        
        if source_ratio > target_ratio_val:
            # 來源比目標寬 (常見的橫轉直案例) -> 以高度為基準，裁切寬度
            crop_h = src_h
            crop_w = int(src_h * target_ratio_val)
        else:
            # 來源比目標窄或一樣 (直轉直、或是更窄的影片) -> 以寬度為基準，裁切高度
            crop_w = src_w
            crop_h = int(src_w / target_ratio_val)

        # 確保為偶數且不超過原始解析度
        crop_w = min(crop_w - crop_w % 2, src_w)
        crop_h = min(crop_h - crop_h % 2, src_h)
        crop_x, crop_y = (src_w - crop_w) // 2, (src_h - crop_h) // 2

        f_w, f_h = self.config.final_ratio
        final_w, final_h = crop_w, int(crop_w * f_h / f_w)
        final_w += final_w % 2  # 確保寬度為偶數
        final_h += final_h % 2  # 確保高度為偶數

        pad_top = max(0, (final_h - crop_h) // 2)
        pad_bottom = max(0, (final_h - crop_h) - pad_top)

        return {
            "crop_w": crop_w, "crop_h": crop_h, "crop_x": crop_x, "crop_y": crop_y,
            "pad_top": pad_top, "pad_bottom": pad_bottom, "final_w": final_w, "final_h": final_h
        }

    # 已移除 get_resolutions_to_process，邏輯移至 process_single_video 中



    def build_ffmpeg_split_command(self, input_file, dims, resolutions_map, info):
        """利用 FFmpeg -filter_complex 實作單次解碼多路輸出"""
        cmd = [self.config.ffmpeg_path, "-hide_banner", "-y"]
        
        # 偵測是否所有輸出的 hwaccel 是一致的
        hwaccels = set()
        for entry in resolutions_map:
            vcodec = entry[5] if len(entry) > 5 else "h265"
            hw = self.h265_hwaccel if vcodec == "h265" else self.h264_hwaccel
            if hw:
                hwaccels.add(hw)
        
        if len(hwaccels) == 1:
            hw = next(iter(hwaccels))
            # VideoToolbox 僅做硬體編碼，不加解碼加速：
            # 因為 filter_complex (drawtext 等) 為 CPU filter，
            # 若問时啟用 videotoolbox 解碼，影格會留在 GPU 記懶體而導致適型錯誤。
            if hw != "videotoolbox":
                cmd += ["-hwaccel", hw]
        
        cmd += ["-i", str(input_file)]

        filters = []
        crop = f"crop={dims['crop_w']}:{dims['crop_h']}:{dims['crop_x']}:{dims['crop_y']}"
        pad = f"pad={dims['final_w']}:{dims['final_h']}:0:{dims['pad_top']}:black" if dims['pad_top'] > 0 or dims['pad_bottom'] > 0 else ""

        # 共同解碼基底
        base = f"[0:v]{crop}[crp]"
        if pad:
            base += f";[crp]{pad}[pad]"
            root_lbl = "[pad]"
        else:
            root_lbl = "[crp]"
        
        # 依解析度分組 (out_w, out_h)
        res_groups = OrderedDict()
        for i, entry in enumerate(resolutions_map):
            key = (entry[0], entry[1]) # (w, h)
            res_groups.setdefault(key, []).append(i)

        splits_cnt = len(res_groups)
        if splits_cnt > 1:
            split_lbls = "".join([f"[base_{i}]" for i in range(splits_cnt)])
            base += f";{root_lbl}split={splits_cnt}{split_lbls}"
            base_inputs = [f"[base_{i}]" for i in range(splits_cnt)]
        else:
            base_inputs = [root_lbl]
            
        filters.append(base)

        font_path = str(self.script_dir / self.config.font_path).replace("\\", "/").replace(":", "\\:")
        top_txt = self.config.top_text_content
        btm_txt = self.config.bottom_text_content

        final_video_maps = [None] * len(resolutions_map)
        
        # 處理每個解析度群組
        for j, ((out_w, out_h), indices) in enumerate(res_groups.items()):
            # 第一步：先做 Scale 和 Drawtext 至一個解析度共通標籤
            res_lbl = f"[res_{j}]"
            seq = f"{base_inputs[j]}scale={out_w}:{out_h}:flags=lanczos{res_lbl}"
            curr_lbl = res_lbl

            scale_rate = out_h / 1920.0
            border_c = self.config.font_color

            if top_txt:
                fz = int(self.config.top_font_size * scale_rate)
                bw = max(1, int(fz * 0.03))
                mar = int(self.config.text_margin * scale_rate)
                ptop = int(dims["pad_top"] * (out_h / dims["final_h"]))
                lines = top_txt.splitlines()
                
                for ln_i, ln in enumerate(lines):
                    esc = ln.replace("'", "'\\''" ).replace(":", "\\:").replace("%", "%%")
                    rev_i = len(lines) - 1 - ln_i
                    y_pos = f"{ptop}-{mar}-text_h-{rev_i}*line_h*{self.config.text_line_spacing}"
                    next_lbl = f"[t_{j}_{ln_i}]"
                    seq += f";{curr_lbl}drawtext=fontfile='{font_path}':text='{esc}':fontsize={fz}:" \
                           f"fontcolor={self.config.font_color}:borderw={bw}:bordercolor={border_c}:" \
                           f"x=(w-text_w)/2:y={y_pos}{next_lbl}"
                    curr_lbl = next_lbl

            if btm_txt:
                fz = int(self.config.bottom_font_size * scale_rate)
                bw = max(1, int(fz * 0.03))
                mar = int(self.config.text_margin * scale_rate)
                pbtm = int(dims["pad_bottom"] * (out_h / dims["final_h"]))
                lines = btm_txt.splitlines()
                
                for ln_i, ln in enumerate(lines):
                    esc = ln.replace("'", "'\\''" ).replace(":", "\\:").replace("%", "%%")
                    y_pos = f"{out_h}-{pbtm}+{mar}+{ln_i}*line_h*{self.config.text_line_spacing}"
                    next_lbl = f"[b_{j}_{ln_i}]"
                    seq += f";{curr_lbl}drawtext=fontfile='{font_path}':text='{esc}':fontsize={fz}:" \
                           f"fontcolor={self.config.font_color}:borderw={bw}:bordercolor={border_c}:" \
                           f"x=(w-text_w)/2:y={y_pos}{next_lbl}"
                    curr_lbl = next_lbl

            # 第二步：將處理完的解析度共通標籤 split 給各個 codec
            num_codecs = len(indices)
            if num_codecs > 1:
                codec_split_lbls = "".join([f"[out_{idx}]" for idx in indices])
                seq += f";{curr_lbl}split={num_codecs}{codec_split_lbls}"
                for idx in indices:
                    final_video_maps[idx] = f"[out_{idx}]"
            else:
                final_video_maps[indices[0]] = curr_lbl
            
            filters.append(seq)
            
        cmd += ["-filter_complex", ";".join(filters)]

        for i, entry in enumerate(resolutions_map):
            out_w, out_h, label, vbr, out_file = entry[:5]
            vcodec = entry[5] if len(entry) > 5 else "h265"
            
            cmd += ["-map", final_video_maps[i]]
            if info["has_audio"]: cmd += ["-map", "0:a:0"]
            
            v_tag = "v:0"
            encoder = self.h265_encoder if vcodec == "h265" else self.h264_encoder
            
            # 根據不同編碼器套用參數 (參考 auto_compress.py)
            if encoder in ["hevc_nvenc", "h264_nvenc"]:
                cmd += [f"-c:{v_tag}", encoder, f"-b:{v_tag}", vbr,
                        "-preset", "p4", "-rc", "vbr"]
            elif encoder in ["hevc_amf", "h264_amf"]:
                cmd += [f"-c:{v_tag}", encoder, f"-b:{v_tag}", vbr,
                        "-quality", "balanced", "-rc", "vbr_latency"]
            elif encoder in ["hevc_qsv", "h264_qsv"]:
                cmd += [f"-c:{v_tag}", encoder, f"-b:{v_tag}", vbr,
                        "-preset", "medium"]
            elif encoder in ["hevc_videotoolbox", "h264_videotoolbox"]:
                # Apple VideoToolbox (macOS): 支援 Apple Silicon 及 Intel Mac GPU 硬體加速
                # -allow_sw 1: 當硬體不可用時自動回退至軟體編碼（保险）
                # -realtime 0: 關閉即時模式，提升編碼品質
                cmd += [f"-c:{v_tag}", encoder, f"-b:{v_tag}", vbr,
                        "-allow_sw", "1", "-realtime", "0"]
            else:
                # libx265 / libx264
                cmd += [f"-c:{v_tag}", encoder,
                        "-preset", "medium", f"-b:{v_tag}", vbr,
                        f"-maxrate:{v_tag}", double_bitrate(vbr),
                        f"-bufsize:{v_tag}", double_bitrate(vbr)]
            
            cmd += ["-pix_fmt", "yuv420p"]
            if vcodec == "h265":
                cmd += ["-tag:v", "hvc1"]
                
            if info["has_audio"]:
                a_tag = "a:0"
                cmd += [f"-c:{a_tag}", "aac", f"-b:{a_tag}", "192k"]
                
            cmd += ["-f", "mp4", "-movflags", "+faststart", str(out_file)]
            
        return cmd

    def process_single_video(self, task_info: Tuple[int, int, Path]) -> bool:
        idx, total, file_path = task_info
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info: return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 將 targets 依照 ratio 分群，以減少重複的 crop 濾鏡運算
        ratio_groups = OrderedDict()
        for t in self.config.targets:
            ratio_groups.setdefault(t['ratio'], []).append(t)

        all_success = True
        for (rt_w, rt_h), targets in ratio_groups.items():
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))
            
            # 過濾並封裝本次比例下所有的輸出任務 (可能包含不同解析度與編碼器)
            active_maps = []
            tmps = []
            finals = []
            
            for t in targets:
                res_key = t['resolution'].lower()
                vcodec = t['vcodec']
                
                # 1. 取得目標短邊上限
                target_short_limit = RESOLUTION_MAP.get(res_key, 1080)
                
                # 2. 計算目前比例下的「來源短邊」
                f_w, f_h = self.config.final_ratio
                source_short = min(dims['final_w'], dims['final_h'])
                
                # 決定「不大於上限」且「不大於來源」的最高標準短邊
                upper_bound_short = min(target_short_limit, source_short)
                
                # 標準短邊清單 (由高到低)
                standard_shorts = [2160, 1440, 1080, 720, 480, 360]
                
                final_short = upper_bound_short # 預設
                for s in standard_shorts:
                    if s <= upper_bound_short:
                        final_short = s
                        break
                
                # 3. 根據 final_short 換算出最終高度 (out_h) 與寬度 (out_w)
                # 以 9:16 為例，短邊為寬，長邊為高
                if f_w <= f_h:
                    out_w = final_short
                    out_h = int(final_short * f_h / f_w)
                else:
                    out_h = final_short
                    out_w = int(final_short * f_w / f_h)

                # 確保皆為偶數
                out_w += out_w % 2
                out_h += out_h % 2
                
                # 決定標籤顯示 (依據最終短邊來決定標籤，最準確)
                if final_short >= 2160: lbl = "4K"
                elif final_short >= 1440: lbl = "2K"
                elif final_short >= 1080: lbl = "FHD"
                elif final_short >= 720: lbl = "HD"
                else: lbl = f"{final_short}P"
                
                suffix_name = f"{rt_w}x{rt_h}_{lbl}_{vcodec}"
                sub_dir = out_dir / suffix_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                target_f = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"
                
                if self.config.skip_existing and target_f.exists():
                    continue

                tmp_f = target_f.with_name(target_f.name + ".tmp")
                bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
                
                # 封裝進 active_maps: (寬, 高, 標籤, 位元率, 暫存檔, 編碼器)
                active_maps.append((out_w, out_h, lbl, bitrate, tmp_f, vcodec))
                tmps.append(tmp_f)
                finals.append(target_f)

            if not active_maps:
                continue

            cmd = self.build_ffmpeg_split_command(file_path, dims, active_maps, info)
            
            # --- 利用 Popen 捕捉進度 ---
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                    universal_newlines=True, encoding='utf-8', errors='replace')
            
            desc = f"({idx}/{total}) {file_path.stem[:12]} [{rt_w}:{rt_h}]"
            pbar = None
            pos = self.position_q.get() if hasattr(self, 'position_q') else 0
            if HAS_TQDM:
                pbar = tqdm(total=info["duration"], desc=desc, position=pos, leave=False,
                            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {elapsed}<{remaining}")
            else:
                sys.stdout.write(f"\n{desc} 處理中...")
                sys.stdout.flush()

            stderr_log = []
            debug_log_path = None
            if self.config.debug:
                debug_log_path = self.script_dir / f"ffmpeg_debug_{file_path.stem}_{rt_w}x{rt_h}.log"

            debug_fd = None
            try:
                if debug_log_path:
                    debug_fd = open(debug_log_path, "w", encoding="utf-8")
                    debug_fd.write(f"[{file_path.name} - {rt_w}:{rt_h}]\n"
                                   f"{' '.join(shlex.quote(s) for s in cmd)}\n\n")

                for line in proc.stdout:
                    stderr_log.append(line)
                    if len(stderr_log) > 15: stderr_log.pop(0)
                    
                    if debug_fd:
                        debug_fd.write(line)
                        debug_fd.flush()
                    
                    if 'time=' in line:
                        match = re.search(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})', line)
                        if match and pbar:
                            sec = parse_ffmpeg_time(match.group(1))
                            pbar.n = min(sec, info["duration"])
                            pbar.refresh()
                
                proc.wait()
            except Exception:
                proc.terminate()
                proc.wait()
                raise
            finally:
                if pbar: pbar.close()
                if debug_fd: debug_fd.close()
                if hasattr(self, 'position_q'): self.position_q.put(pos)

            if proc.returncode != 0:
                if HAS_TQDM:
                    tqdm.write(f" [失敗!] ({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}]")
                else:
                    print(" [失敗!]")
                print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
                print(f"指令輸出結尾：\n{''.join(stderr_log)}")
                for t in tmps:
                    if t.exists(): t.unlink()
                all_success = False
                continue

            if HAS_TQDM:
                tqdm.write(f"({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}] 完成!")
            else:
                print(" [完成!]")

            for t, f in zip(tmps, finals):
                if t.exists():
                    if f.exists(): f.unlink()
                    t.rename(f)

        return all_success

    def run(self):
        in_dir = Path(self.config.input_dir)
        if not in_dir.exists():
            in_dir.mkdir(parents=True)
            print(f"\n[提示] 未找到 '{in_dir.resolve()}'，已自動創建，請放置影片後重新執行。")
            return

        # 清理先前殘留的 .tmp 暫存檔
        out_dir = Path(self.config.output_dir)
        cleanup_tmp_files(out_dir)
            
        videos = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in self.config.video_extensions]
        if not videos:
            print(f"\n[提示] 資料夾內無可支援的影片檔。")
            return

        # 若 max_workers 設為 0，自動以系統核心數的一半作為基準，但最高限制為 4，避免 Bug 或過載
        workers = self.config.max_workers
        if workers <= 0:
            workers = (os.cpu_count() or 2) // 2
        
        # 強制限制最大平行數為 4
        if workers > 4:
            workers = 4
        if workers < 1:
            workers = 1

        self.position_q = Queue()
        for i in range(workers):
            self.position_q.put(i)

        print(f"\n找到 {len(videos)} 個目標將開始轉換 (平行任務數: {workers})...\n")

        success_count, failed_files = 0, []
        tasks = [(i, len(videos), v) for i, v in enumerate(sorted(videos), 1)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.process_single_video, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    if fut.result(): success_count += 1
                    else: failed_files.append(t[2].name)
                except Exception as e:
                    failed_files.append(t[2].name)
                    print(f"\n[錯誤] 處理 {t[2].name} 時發生異常: {e}")

        print("\n" + "=" * 60)
        print("  任務總結")
        print(f"  成功: {success_count} / {len(videos)}")
        if failed_files:
            print(f"  失敗: {', '.join(failed_files)}")
        print("=" * 60)


def main():
    print("=" * 60)
    print("  Auto Reframe Video v2.0 - H.265 高效能優化版")
    print("=" * 60)
    
    config = ReframeConfig()
    app = VideoReframer(config)
    app.run()
    
    # 執行完畢後暫停，避免視窗直接關閉
    if os.name == 'nt':
        os.system("pause")
    else:
        input("\n請按 Enter 鍵結束...")


if __name__ == "__main__":
    main()
