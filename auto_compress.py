# -*- coding: utf-8 -*-
"""
Auto Compress Video 
將輸入資料夾的影片，依照 YouTube 建議的位元率壓縮出所有可能解析度 (4K, 2K, FHD)。
共用核心邏輯包含在 video_utils.py 之中。
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_utils import (
    detect_hw_encoder, get_video_info, double_bitrate,
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


@dataclass
class CompressConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    video_extensions: set = field(default_factory=lambda: {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"})
    skip_existing: bool = True
    max_workers: int = 0
    debug: bool = False


class VideoCompressor:
    def __init__(self, config: CompressConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent
        self.encoder, self.hwaccel = detect_hw_encoder(self.config.ffmpeg_path)

    def get_compress_tiers(self, info: dict) -> List[Tuple[int, int, str, str]]:
        tiers = []
        w, h = info["width"], info["height"]
        short_side = min(w, h)
        is_vert = h > w
        
        def calc_dims(max_short, max_long):
            tgt_w = max_short if is_vert else max_long
            tgt_h = max_long if is_vert else max_short
            
            if w > 0 and h > 0:
                scale_w = tgt_w / w
                scale_h = tgt_h / h
                scale = min(scale_w, scale_h)
                if scale > 1: scale = 1.0  # 不升頻
                new_w = int(w * scale)
                new_h = int(h * scale)
                new_w += new_w % 2
                new_h += new_h % 2
                return new_w, new_h
            return w, h
            
        def add_tier(tgt_short, tgt_long, label):
            cw, ch = calc_dims(tgt_short, tgt_long)
            bitrate = get_youtube_bitrate(min(cw, ch), info["fps"])
            tiers.append((cw, ch, label, bitrate))

        # 根據短邊判定最高層級
        if short_side >= 2160 * 0.9:
            add_tier(2160, 3840, "COMPRESS_4K")
        if short_side >= 1440 * 0.9:
            add_tier(1440, 2560, "COMPRESS_2K")
        if short_side >= 1080 * 0.9:
            add_tier(1080, 1920, "COMPRESS_FHD")
            
        # 小於 FHD 則以原始尺寸壓縮
        if not tiers:
            new_w, new_h = w - w % 2, h - h % 2
            bitrate = get_youtube_bitrate(short_side, info["fps"])
            tiers.append((new_w, new_h, "COMPRESS_Original", bitrate))
            
        return tiers

    def build_ffmpeg_split_command(self, input_file: Path, tiers_map: list, info: dict) -> list:
        cmd = [self.config.ffmpeg_path, "-hide_banner", "-y"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
        cmd += ["-i", str(input_file)]

        filters = []
        splits_cnt = len(tiers_map)
        if splits_cnt > 1:
            split_lbls = "".join([f"[base_{i}]" for i in range(splits_cnt)])
            filters.append(f"[0:v]split={splits_cnt}{split_lbls}")
            base_inputs = [f"[base_{i}]" for i in range(splits_cnt)]
        else:
            base_inputs = ["[0:v]"]
            
        final_video_maps = []
        for i, (out_w, out_h, label, vbr, out_file) in enumerate(tiers_map):
            scale_lbl = f"[out_{i}]"
            # 單純壓縮縮放，不加任何濾鏡
            seq = f"{base_inputs[i]}scale={out_w}:{out_h}:flags=lanczos{scale_lbl}"
            filters.append(seq)
            final_video_maps.append(scale_lbl)
            
        cmd += ["-filter_complex", ";".join(filters)]

        for i, (out_w, out_h, label, vbr, out_file) in enumerate(tiers_map):
            cmd += ["-map", final_video_maps[i]]
            if info["has_audio"]: cmd += ["-map", "0:a:0"]
            
            v_tag = "v:0"
            if self.encoder == "h264_nvenc":
                cmd += [f"-c:{v_tag}", self.encoder, f"-b:{v_tag}", vbr,
                        "-preset", "p4", "-rc", "vbr", "-cq", "20"]
            elif self.encoder == "h264_amf":
                cmd += [f"-c:{v_tag}", self.encoder, f"-b:{v_tag}", vbr,
                        "-quality", "balanced", "-rc", "vbr_latency"]
            elif self.encoder == "h264_qsv":
                cmd += [f"-c:{v_tag}", self.encoder, f"-b:{v_tag}", vbr,
                        "-preset", "medium", "-global_quality", "22"]
            else:
                cmd += [f"-c:{v_tag}", self.encoder,
                        "-preset", "medium", "-crf", "18",
                        f"-maxrate:{v_tag}", vbr,
                        f"-bufsize:{v_tag}", double_bitrate(vbr)]
            
            cmd += ["-pix_fmt", "yuv420p"]
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
        
        tiers = self.get_compress_tiers(info)
        active_maps = []
        tmps = []
        finals = []
        
        for w, h, lbl, bitrate in tiers:
            sub_dir = out_dir / lbl
            sub_dir.mkdir(parents=True, exist_ok=True)
            # 檔案名稱：原檔片名_COMPRESS_解析度.mp4，例如 test_COMPRESS_4K.mp4
            target_f = sub_dir / f"{file_path.stem}_{lbl}.mp4"
            if self.config.skip_existing and target_f.exists(): continue
            tmp_f = target_f.with_name(target_f.name + ".tmp")
            
            active_maps.append((w, h, lbl, bitrate, tmp_f))
            tmps.append(tmp_f)
            finals.append(target_f)

        if not active_maps:
            return True

        cmd = self.build_ffmpeg_split_command(file_path, active_maps, info)
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                universal_newlines=True, encoding='utf-8', errors='replace')
        
        desc = f"({idx}/{total}) {file_path.stem[:12]} [Auto Compress]"
        pbar = None
        if HAS_TQDM:
            pbar = tqdm(total=info["duration"], desc=desc, position=0, leave=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {elapsed}<{remaining}")
        else:
            sys.stdout.write(f"\n{desc} 處理中...")
            sys.stdout.flush()

        stderr_log = []
        debug_log_path = None
        if self.config.debug:
            debug_log_path = self.script_dir / f"ffmpeg_debug_{file_path.stem}_compress.log"

        debug_fd = None
        try:
            if debug_log_path:
                debug_fd = open(debug_log_path, "w", encoding="utf-8")
                debug_fd.write(f"[{file_path.name}]\n"
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

        return True

    def run(self):
        in_dir = Path(self.config.input_dir)
        if not in_dir.exists():
            in_dir.mkdir(parents=True)
            print(f"\n[提示] 未找到 '{in_dir.resolve()}'，已自動創建，請放置影片後重新執行。")
            return

        out_dir = Path(self.config.output_dir)
        cleanup_tmp_files(out_dir)
            
        videos = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in self.config.video_extensions]
        if not videos:
            print(f"\n[提示] 資料夾內無可支援的影片檔。")
            return

        workers = self.config.max_workers
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) // 2)

        print(f"\n找到 {len(videos)} 個目標將開始進行多重解析度壓縮 (平行任務數: {workers})...\n")

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
    print("  Auto Compress Video - 多重解析度壓縮工具")
    print("=" * 60)
    
    config = CompressConfig()
    app = VideoCompressor(config)
    app.run()
    
    if os.name == 'nt':
        os.system("pause")
    else:
        input("\n請按 Enter 鍵結束...")

if __name__ == "__main__":
    main()
