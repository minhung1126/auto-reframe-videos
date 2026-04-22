# -*- coding: utf-8 -*-
"""
Auto Compress Video (H.264 / H.265)
將輸入資料夾的影片，依據原始畫質選擇適當的 YouTube 建議位元率壓縮單一解析度。
同時輸出 H.265 (HEVC) 與 H.264 (AVC) 兩種版本，採用單次解碼 + filter_complex split 以提高效能。
"""

import os
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple
from queue import Queue

from video_utils import (
    # 硬體加速偵測
    detect_h265_hw_encoder, detect_h264_hw_encoder,
    # 影片資訊
    get_video_info,
    # 位元率工具
    get_youtube_bitrate, double_bitrate,
    # 暫存清理
    cleanup_tmp_files,
    # 共用常數
    h264, h265, RESOLUTION_MAP,
    # 解析度工具
    resolve_short_side, resolution_label,
    # 執行工具
    resolve_workers, detect_hwaccel_for_cmd,
    build_output_args,
    tqdm_write, run_ffmpeg_with_progress, run_parallel,
)

# 強制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
    video_extensions: set = field(default_factory=lambda: {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"
    })
    # 是否跳過已存在的輸出檔案
    skip_existing: bool = True
    # 平行處理的任務數量，設為 0 將自動判斷 (macOS 上限為 8)
    max_workers: int = 0
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

        # 收集所有輸出的 hwaccel 後端，用共用工具決定是否啟用解碼端加速
        hwaccels = set()
        for _, _, _, _, _, vcodec in tiers_map:
            hw = self.h265_hwaccel if vcodec == h265 else self.h264_hwaccel
            if hw:
                hwaccels.add(hw)
        cmd += detect_hwaccel_for_cmd(hwaccels)

        cmd += ["-i", str(input_file)]

        # 依解析度分組：相同解析度只需 scale 一次，再 split 給各 codec
        res_groups: OrderedDict = OrderedDict()
        for i, entry in enumerate(tiers_map):
            key = (entry[0], entry[1])  # (out_w, out_h)
            res_groups.setdefault(key, []).append(i)

        num_resolutions = len(res_groups)
        filters = []
        final_video_maps = [None] * len(tiers_map)

        if num_resolutions > 1:
            split_lbls = "".join([f"[raw_{j}]" for j in range(num_resolutions)])
            filters.append(f"[0:v]split={num_resolutions}{split_lbls}")
            raw_inputs = [f"[raw_{j}]" for j in range(num_resolutions)]
        else:
            raw_inputs = ["[0:v]"]

        for j, ((out_w, out_h), indices) in enumerate(res_groups.items()):
            num_codecs = len(indices)
            # 只有當解析度不同時才進行 scale，避免不必要的 GPU-CPU 搬運
            if out_w == info["width"] and out_h == info["height"]:
                curr_input = raw_inputs[j]
            else:
                scaled_lbl = f"[scaled_{j}]"
                filters.append(f"{raw_inputs[j]}scale={out_w}:{out_h}:flags=bicubic{scaled_lbl}")
                curr_input = scaled_lbl

            if num_codecs > 1:
                codec_split_lbls = "".join([f"[out_{idx}]" for idx in indices])
                filters.append(f"{curr_input}split={num_codecs}{codec_split_lbls}")
                for idx in indices:
                    final_video_maps[idx] = f"[out_{idx}]"
            else:
                final_video_maps[indices[0]] = curr_input

        if filters:
            cmd += ["-filter_complex", ";".join(filters)]

        for i, (out_w, out_h, label, vbr, out_file, vcodec) in enumerate(tiers_map):
            v_map = final_video_maps[i]
            # 如果沒有使用 filter_complex，將標籤 [0:v] 轉為直接映射 0:v
            if not filters and v_map.startswith("[") and v_map.endswith("]"):
                v_map = v_map[1:-1]
            
            cmd += ["-map", v_map]
            if info["has_audio"]:
                cmd += ["-map", "0:a:0"]
            encoder = self.h265_encoder if vcodec == h265 else self.h264_encoder
            cmd += build_output_args(encoder, vbr, vcodec, info["has_audio"], out_file)

        return cmd

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info:
            return False

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

            final_short = resolve_short_side(res_key, source_short)
            scale_factor = min(final_short / source_short, 1.0) if source_short > 0 else 1.0

            out_w = int(src_w * scale_factor)
            out_h = int(src_h * scale_factor)
            out_w += out_w % 2
            out_h += out_h % 2

            lbl = f"COMPRESS_{resolution_label(final_short)}"
            suffix_name = f"{lbl}_{vcodec}"
            sub_dir = out_dir / suffix_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            target_f = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

            if self.config.skip_existing and target_f.exists():
                continue

            tmp_f = target_f.with_name(target_f.name + ".tmp")
            bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
            active_maps.append((out_w, out_h, lbl, bitrate, tmp_f, vcodec))
            tmps.append(tmp_f)
            finals.append(target_f)

        if not active_maps:
            return True

        cmd = self.build_ffmpeg_split_command(file_path, active_maps, info)

        debug_log_path = None
        if self.config.debug:
            debug_log_path = self.script_dir / f"ffmpeg_debug_{file_path.stem}_compress.log"

        desc = f"({idx}/{total}) {file_path.stem[:12]} [Auto Compress]"
        returncode, stderr_log = run_ffmpeg_with_progress(
            cmd, info, desc, position_q, debug_log_path
        )

        if returncode != 0:
            tqdm_write(f" [失敗!] ({idx}/{total}) {file_path.name}")
            print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
            print(f"完整執行指令：\n{' '.join(shlex.quote(s) for s in cmd)}")
            print(f"\n指令輸出結尾：\n{''.join(stderr_log)}")
            for t in tmps:
                if t.exists():
                    t.unlink()
            return False

        tqdm_write(f"({idx}/{total}) {file_path.name} 處理完成!")

        for t, f in zip(tmps, finals):
            if t.exists():
                if f.exists():
                    f.unlink()
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

        videos = [
            f for f in in_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.video_extensions
        ]
        if not videos:
            print("\n[提示] 資料夾內無可支援的影片檔。")
            return

        workers = resolve_workers(self.config.max_workers)
        print(f"\n找到 {len(videos)} 個目標將開始進行自動壓縮 (平行任務數: {workers})...\n")

        tasks = [(i, len(videos), v) for i, v in enumerate(sorted(videos), 1)]
        success_count, failed_files = run_parallel(tasks, self.process_single_video, workers)

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
