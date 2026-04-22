# -*- coding: utf-8 -*-
"""
Auto Compress Video (H.264 / H.265)
將輸入資料夾的影片，依據原始畫質選擇適當的 YouTube 建議位元率壓縮單一解析度。
同時輸出 H.265 (HEVC) 與 H.264 (AVC) 兩種版本，採用單次解碼 + filter_complex split 以提高效能。
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
from queue import Queue

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
h264 = "h264"
h265 = "h265"

# 解析度關鍵字對應表 (指定短邊長度)
RESOLUTION_MAP = {
    "4k": 2160,
    "2k": 1440,
    "1080p": 1080,
    "fhd": 1080,
    "720p": 720,
    "hd": 720,
    "480p": 480,
    "360p": 360,
    "source": 99999  # 代表不縮放，直接取原始短邊
}

@dataclass
class CompressConfig:
    # --- 輸入 / 輸出 ---
    # 輸入影片的資料夾路徑
    input_dir: str = "input"
    # 輸出影片的資料夾路徑
    output_dir: str = "output"
    
    # --- 輸出目標 (重要設定) ---
    # resolution: '4k', '2k', '1080p', 'fhd', '720p', 'hd', '480p', '360p', 'source'
    # vcodec: h264, h265
    targets: List[dict] = field(default_factory=lambda: [
        {'resolution': 'source', 'vcodec': h265},
        # {'resolution': '1080p', 'vcodec': h264},
    ])

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
    # 是否開啟除錯模式
    debug: bool = False


class VideoCompressor:
    def __init__(self, config: CompressConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent
        self.h265_encoder, self.h265_hwaccel = detect_h265_hw_encoder(self.config.ffmpeg_path)
        self.h264_encoder, self.h264_hwaccel = detect_h264_hw_encoder(self.config.ffmpeg_path)


    def build_ffmpeg_split_command(self, input_file: Path, tiers_map: list, info: dict) -> list:
        cmd = [self.config.ffmpeg_path, "-hide_banner", "-y"]

        # 只有當所有 codec 的 hwaccel 後端一致時才啟用，避免廠商不同導致衝突
        hwaccels = set()
        for _, _, _, _, _, vcodec in tiers_map:
            hw = self.h265_hwaccel if vcodec == "h265" else self.h264_hwaccel
            if hw:
                hwaccels.add(hw)
        if len(hwaccels) == 1:
            hw = next(iter(hwaccels))
            # VideoToolbox 僅做硬體編碼，不加解碼加速：
            # 即使沒有 drawtext 漿鏡，丿統一與 auto_reframe.py 行為一致。
            if hw != "videotoolbox":
                cmd += ["-hwaccel", hw]

        cmd += ["-i", str(input_file)]

        # 依解析度分組：相同解析度只需 scale 一次，再 split 給各 codec
        # 例：4K × {h265, h264} → scale 一次 → split=2 → 各自編碼
        from collections import OrderedDict
        res_groups: OrderedDict = OrderedDict()
        for i, entry in enumerate(tiers_map):
            key = (entry[0], entry[1])  # (out_w, out_h)
            res_groups.setdefault(key, []).append(i)

        num_resolutions = len(res_groups)
        filters = []
        final_video_maps = [None] * len(tiers_map)

        # 若有多個不同解析度，先 split 原始串流
        if num_resolutions > 1:
            split_lbls = "".join([f"[raw_{j}]" for j in range(num_resolutions)])
            filters.append(f"[0:v]split={num_resolutions}{split_lbls}")
            raw_inputs = [f"[raw_{j}]" for j in range(num_resolutions)]
        else:
            raw_inputs = ["[0:v]"]

        # 每個解析度 scale 一次，再依 codec 數量決定是否再 split
        for j, ((out_w, out_h), indices) in enumerate(res_groups.items()):
            num_codecs = len(indices)
            if num_codecs > 1:
                # scale → split → 各 codec（核心效能優化：相同解析度只縮放一次）
                scaled_lbl = f"[scaled_{j}]"
                filters.append(f"{raw_inputs[j]}scale={out_w}:{out_h}:flags=lanczos{scaled_lbl}")
                codec_split_lbls = "".join([f"[out_{idx}]" for idx in indices])
                filters.append(f"{scaled_lbl}split={num_codecs}{codec_split_lbls}")
            else:
                # 只有單一 codec，直接 scale 到輸出 label
                idx = indices[0]
                filters.append(f"{raw_inputs[j]}scale={out_w}:{out_h}:flags=lanczos[out_{idx}]")

            for idx in indices:
                final_video_maps[idx] = f"[out_{idx}]"

        cmd += ["-filter_complex", ";".join(filters)]

        for i, (out_w, out_h, label, vbr, out_file, vcodec) in enumerate(tiers_map):
            cmd += ["-map", final_video_maps[i]]
            if info["has_audio"]: cmd += ["-map", "0:a:0"]

            v_tag = "v:0"
            encoder = self.h265_encoder if vcodec == "h265" else self.h264_encoder

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
                # libx265 / libx264: 使用 YouTube 建議位元率進行平均位元率編碼 (ABR)
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
        
        active_maps = []
        tmps = []
        finals = []
        
        src_w, src_h = info["width"], info["height"]
        source_short = min(src_w, src_h)

        for t in self.config.targets:
            res_key = t['resolution'].lower()
            vcodec = t['vcodec']
            
            target_short_limit = RESOLUTION_MAP.get(res_key, 1080)
            upper_bound_short = min(target_short_limit, source_short)
            standard_shorts = [2160, 1440, 1080, 720, 480, 360]
            
            final_short = upper_bound_short
            for s in standard_shorts:
                if s <= upper_bound_short:
                    final_short = s
                    break

            # 根據 final_short 換算出最終解析度
            scale_factor = final_short / source_short if source_short > 0 else 1.0
            if scale_factor > 1.0: scale_factor = 1.0
            
            out_w = int(src_w * scale_factor)
            out_h = int(src_h * scale_factor)
            
            out_w += out_w % 2
            out_h += out_h % 2

            if final_short >= 2160: lbl = "4K"
            elif final_short >= 1440: lbl = "2K"
            elif final_short >= 1080: lbl = "FHD"
            elif final_short >= 720: lbl = "HD"
            else: lbl = f"{final_short}P"

            lbl = f"COMPRESS_{lbl}"
            suffix_name = f"{lbl}_{vcodec}"
            sub_dir = out_dir / suffix_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            target_f = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"
            
            if self.config.skip_existing and target_f.exists(): continue
            tmp_f = target_f.with_name(target_f.name + ".tmp")
            
            bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
            active_maps.append((out_w, out_h, lbl, bitrate, tmp_f, vcodec))
            tmps.append(tmp_f)
            finals.append(target_f)

        if not active_maps:
            return True

        cmd = self.build_ffmpeg_split_command(file_path, active_maps, info)
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                universal_newlines=True, encoding='utf-8', errors='replace')
        
        desc = f"({idx}/{total}) {file_path.stem[:12]} [Auto Compress]"
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
            if hasattr(self, 'position_q'): self.position_q.put(pos)

        if proc.returncode != 0:
            if HAS_TQDM:
                tqdm.write(f" [失敗!] ({idx}/{total}) {file_path.name}")
            else:
                print(" [失敗!]")
            print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
            print(f"指令輸出結尾：\n{''.join(stderr_log)}")
            for t in tmps:
                if t.exists(): t.unlink()
            return False

        if HAS_TQDM:
            tqdm.write(f"({idx}/{total}) {file_path.name} 處理完成!")
        else:
            print(" [完成!]")

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
            workers = (os.cpu_count() or 2) // 2
            
        # 限制最大平行任務數為 4
        if workers > 4:
            workers = 4
        if workers < 1:
            workers = 1

        self.position_q = Queue()
        for i in range(workers):
            self.position_q.put(i)

        print(f"\n找到 {len(videos)} 個目標將開始進行自動壓縮 (平行任務數: {workers})...\n")

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
    print("  Auto Compress Video - H.264 / H.265 自動解析度壓縮工具")
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
