# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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


@dataclass
class ReframeConfig:
    # --- 輸入 / 輸出 ---
    input_dir: str = "input"
    output_dir: str = "output"

    # --- 裁切比例 (寬:高) ---
    target_ratios: List[Tuple[int, int]] = field(default_factory=lambda: [(4, 5), (1, 1)])
    final_ratio: Tuple[int, int] = (9, 16)

    # --- 上方文字設定 ---
    top_text_file: str = "top_text.txt"
    top_font_size: int = 48

    # --- 下方文字設定 ---
    bottom_text_file: str = "bottom_text.txt"
    bottom_font_size: int = 24

    # --- 字型設定 ---
    font_path: str = "fonts/NotoSerifTC.ttf"
    font_color: str = "white"
    text_margin: int = 20
    text_line_spacing: float = 1.2

    # --- 系統與平行化 ---
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    video_extensions: set = field(default_factory=lambda: {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"})
    skip_existing: bool = True
    max_workers: int = 1  # 預設同時處理 1 支影片，視硬體效能可調高

    # 執行階段產生，不需手動設定
    top_text_content: str = ""
    bottom_text_content: str = ""


class VideoReframer:
    def __init__(self, config: ReframeConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent

        # 1. 初始化讀取：只讀一次，省去每次迴圈讀檔的 I/O 損耗
        self.load_texts()

        # 2. 偵測可用的硬體加速
        self.encoder, self.hwaccel = self.detect_hw_encoder()

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

    def detect_hw_encoder(self):
        try:
            res = subprocess.run([self.config.ffmpeg_path, "-hide_banner", "-encoders"], 
                                 capture_output=True, text=True, timeout=10)
            encoders = res.stdout
        except Exception:
            print(f"[錯誤] 呼叫 {self.config.ffmpeg_path} 失敗，請確認其是否存在。")
            sys.exit(1)

        # 優先順序: NVENC > AMF > QSV
        for enc, hw in [("h264_nvenc", "cuda"), ("h264_amf", "d3d11va"), ("h264_qsv", "qsv")]:
            if enc in encoders:
                test = subprocess.run(
                    [self.config.ffmpeg_path, "-hide_banner", "-f", "lavfi", "-i", 
                     "nullsrc=s=256x256:d=1", "-c:v", enc, "-f", "null", "-"],
                    capture_output=True, text=True, timeout=15
                )
                if test.returncode == 0:
                    print(f"  [核心系統] 已啟用硬體加速編碼器: {enc} ({hw})")
                    return enc, hw

        print("  [核心系統] 未發現可用硬體加速，回退至軟體編碼 (libx264)")
        return "libx264", None

    def get_video_info(self, input_file: Path):
        try:
            res = subprocess.run(
                [self.config.ffprobe_path, "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", str(input_file)],
                capture_output=True, text=True, timeout=30
            )
            data = json.loads(res.stdout)
        except Exception:
            return None

        v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

        if not v_stream:
            return None

        fps_str = v_stream.get("r_frame_rate", "30/1")
        fps = float(fps_str.split('/')[0]) / float(fps_str.split('/')[1]) if '/' in fps_str and float(fps_str.split('/')[1]) != 0 else float(fps_str)
        
        return {
            "width": int(v_stream.get("width", 0)),
            "height": int(v_stream.get("height", 0)),
            "fps": round(fps, 3),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "has_audio": a_stream is not None,
        }

    def calculate_dimensions(self, src_w, src_h, target_ratio):
        t_w, t_h = target_ratio
        short = min(src_w, src_h)

        if t_w <= t_h:
            crop_h, crop_w = short, int(short * t_w / t_h)
        else:
            crop_w, crop_h = short, int(short * t_h / t_w)

        crop_w = min(crop_w - crop_w % 2, src_w)
        crop_h = min(crop_h - crop_h % 2, src_h)
        crop_x, crop_y = (src_w - crop_w) // 2, (src_h - crop_h) // 2

        f_w, f_h = self.config.final_ratio
        final_w, final_h = crop_w, int(crop_w * f_h / f_w)
        final_h += final_h % 2

        pad_top = max(0, (final_h - crop_h) // 2)
        pad_bottom = max(0, (final_h - crop_h) - pad_top)

        return {
            "crop_w": crop_w, "crop_h": crop_h, "crop_x": crop_x, "crop_y": crop_y,
            "pad_top": pad_top, "pad_bottom": pad_bottom, "final_w": final_w, "final_h": final_h
        }

    def get_resolutions_to_process(self, dims):
        short = min(dims["final_w"], dims["final_h"])
        if short >= 2160: return [(2160, 3840, "4K"), (1080, 1920, "FHD")]
        if short >= 1440: return [(1440, 2560, "2K"), (1080, 1920, "FHD")]
        return [(1080, 1920, "FHD")]

    def select_bitrate(self, out_h, fps):
        high_fps = fps > 30
        if out_h >= 3840: return "60M" if high_fps else "40M"
        if out_h >= 2560: return "24M" if high_fps else "16M"
        return "12M" if high_fps else "8M"

    def parse_ffmpeg_time(self, time_str):
        try:
            h, m, s = time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return 0.0

    def build_ffmpeg_split_command(self, input_file, dims, resolutions_map, info):
        """利用 FFmpeg -filter_complex 實作單次解碼多路輸出"""
        cmd = [self.config.ffmpeg_path, "-hide_banner", "-y"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
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
        
        splits_cnt = len(resolutions_map)
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

        # 建構各輸出流向 (Scale -> Drawtext)
        final_video_maps = []
        for i, (out_w, out_h, label, vbr, out_file) in enumerate(resolutions_map):
            scale_lbl = f"[scl_{i}]"
            seq = f"{base_inputs[i]}scale={out_w}:{out_h}:flags=lanczos{scale_lbl}"
            curr_lbl = scale_lbl

            scale_rate = out_h / 1920.0
            border_c = self.config.font_color

            if top_txt:
                fz = int(self.config.top_font_size * scale_rate)
                bw = max(1, int(fz * 0.03))
                mar = int(self.config.text_margin * scale_rate)
                ptop = int(dims["pad_top"] * (out_h / dims["final_h"]))
                lines = top_txt.splitlines()
                
                for ln_i, ln in enumerate(lines):
                    esc = ln.replace("'", "'\\''").replace(":", "\\:")
                    y_pos = f"{ptop}-{mar}-text_h-({len(lines)-1-ln_i})*line_h*{self.config.text_line_spacing}"
                    next_lbl = f"[t_{i}_{ln_i}]"
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
                    esc = ln.replace("'", "'\\''").replace(":", "\\:")
                    y_pos = f"{out_h}-{pbtm}+{mar}+{ln_i}*line_h*{self.config.text_line_spacing}"
                    next_lbl = f"[b_{i}_{ln_i}]"
                    seq += f";{curr_lbl}drawtext=fontfile='{font_path}':text='{esc}':fontsize={fz}:" \
                           f"fontcolor={self.config.font_color}:borderw={bw}:bordercolor={border_c}:" \
                           f"x=(w-text_w)/2:y={y_pos}{next_lbl}"
                    curr_lbl = next_lbl

            final_video_maps.append(curr_lbl)
            filters.append(seq)
            
        cmd += ["-filter_complex", ";".join(filters)]

        # 指定 Mapping 與編碼參數
        for i, (out_w, out_h, label, vbr, out_file) in enumerate(resolutions_map):
            cmd += ["-map", final_video_maps[i]]
            if info["has_audio"]: cmd += ["-map", "0:a:0"]
            
            v_tag = "v:0"
            cmd += [f"-c:{v_tag}", self.encoder, f"-b:{v_tag}", vbr]
            
            if self.encoder == "h264_nvenc":
                cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
            elif self.encoder == "h264_amf":
                cmd += ["-quality", "balanced", "-rc", "vbr_latency"]
            elif self.encoder == "h264_qsv":
                cmd += ["-preset", "medium", "-global_quality", "22"]
            else:
                cmd += ["-preset", "medium", "-crf", "18", f"-maxrate:{v_tag}", vbr, 
                        f"-bufsize:{v_tag}", f"{int(vbr[:-1])*2}M"]
            
            cmd += ["-pix_fmt", "yuv420p"]
            if info["has_audio"]:
                a_tag = "a:0"
                cmd += [f"-c:{a_tag}", "aac", f"-b:{a_tag}", "192k"]
                
            cmd += ["-f", "mp4", "-movflags", "+faststart", str(out_file)]
            
        return cmd

    def process_single_video(self, task_info: Tuple[int, int, Path]) -> bool:
        idx, total, file_path = task_info
        info = self.get_video_info(file_path)
        if not info: return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        successes = 0

        for rt_w, rt_h in self.config.target_ratios:
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))
            res_tiers = self.get_resolutions_to_process(dims)
            
            # 過濾並封裝本次需要的解析度任務
            active_maps = []
            tmps = []
            finals = []
            
            for (w, h, lbl) in res_tiers:
                target_f = out_dir / f"{file_path.stem}_{rt_w}x{rt_h}_{lbl}.mp4"
                if self.config.skip_existing and target_f.exists(): continue
                tmp_f = target_f.with_name(target_f.name + ".tmp")
                bitrate = self.select_bitrate(h, info["fps"])
                active_maps.append((w, h, lbl, bitrate, tmp_f))
                tmps.append(tmp_f)
                finals.append(target_f)

            if not active_maps:
                successes += 1
                continue

            cmd = self.build_ffmpeg_split_command(file_path, dims, active_maps, info)
            
            # --- 利用 Popen 捕捉進度 ---
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                    universal_newlines=True, encoding='utf-8', errors='replace')
            
            desc = f"({idx}/{total}) {file_path.stem[:12]} [{rt_w}:{rt_h}]"
            pbar = None
            if HAS_TQDM:
                pbar = tqdm(total=info["duration"], desc=desc, position=idx-1, leave=True,
                            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {elapsed}<{remaining}")
            else:
                sys.stdout.write(f"\n{desc} 處理中...")
                sys.stdout.flush()

            stderr_log = []
            for line in proc.stdout:
                stderr_log.append(line)
                if len(stderr_log) > 15: stderr_log.pop(0)
                
                if 'time=' in line:
                    match = re.search(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})', line)
                    if match and pbar:
                        sec = self.parse_ffmpeg_time(match.group(1))
                        pbar.n = min(sec, info["duration"])
                        pbar.refresh()
            
            proc.wait()
            if pbar: pbar.close()

            if proc.returncode != 0:
                if not HAS_TQDM: print(" [失敗!]")
                print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
                print(f"指令輸出結尾：\n{''.join(stderr_log)}")
                for t in tmps:
                    if t.exists(): t.unlink()
                return False

            if not HAS_TQDM: print(" [完成!]")

            for t, f in zip(tmps, finals):
                if t.exists():
                    if f.exists(): f.unlink()
                    t.rename(f)
            successes += 1

        return successes == len(self.config.target_ratios)

    def run(self):
        in_dir = Path(self.config.input_dir)
        if not in_dir.exists():
            in_dir.mkdir(parents=True)
            print(f"\n[提示] 未找到 '{in_dir.resolve()}'，已自動創建，請放置影片後重新執行。")
            return
            
        videos = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in self.config.video_extensions]
        if not videos:
            print(f"\n[提示] 資料夾內無可支援的影片檔。")
            return

        print(f"\n找到 {len(videos)} 個目標將開始轉換 (平行任務數: {self.config.max_workers})...\n")

        success_count, failed_files = 0, []
        tasks = [(i, len(videos), v) for i, v in enumerate(sorted(videos), 1)]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
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
    print("  Auto Reframe Video v2.0 - 高效能優化版")
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
