# -*- coding: utf-8 -*-
"""
Auto Compress Video (H.264 / H.265)
將輸入資料夾的影片，依據原始畫質選擇適當的 YouTube 建議位元率壓縮單一解析度。
同時輸出 H.265 (HEVC) 與 H.264 (AVC) 兩種版本，採用單次解碼 + filter_complex split 以提高效能。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from queue import Queue

from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.ffmpeg_graphs import build_compress_split_command
from auto_reframe_core.output_plans import (
    build_compress_output_plan,
    cleanup_temp_outputs,
    promote_temp_outputs,
)
from auto_reframe_core.platform_profile import (
    pause_for_windows_shell,
    should_pause_with_windows_prompt,
)
from video_utils import (
    # 硬體加速偵測
    detect_h265_hw_encoder, detect_h264_hw_encoder,
    # 影片資訊
    get_video_info,
    # 共用常數
    h264, h265,
    # 執行工具
    tqdm_write, run_ffmpeg_with_progress,
)
from auto_reframe_core.target_specs import normalize_compress_target

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
    # 平行處理的任務數量，設為 0 將自動判斷 (macOS 上限為 4，其他平台上限為 8)
    max_workers: int = 0
    # 是否開啟除錯模式
    debug: bool = False


class VideoCompressor:
    def __init__(self, config: CompressConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent
        self._validate_config()
        self.h265_encoder, self.h265_hwaccel = detect_h265_hw_encoder(self.config.ffmpeg_path)
        self.h264_encoder, self.h264_hwaccel = detect_h264_hw_encoder(self.config.ffmpeg_path)

    def _validate_config(self):
        if not self.config.targets:
            raise ValueError("CompressConfig.targets 不可為空。")

        for idx, target in enumerate(self.config.targets, 1):
            normalize_compress_target(target, idx)

    def build_ffmpeg_split_command(self, input_file: Path, tiers_map: list, info: dict) -> list:
        return build_compress_split_command(
            self.config.ffmpeg_path,
            input_file,
            tiers_map,
            info,
            self.h264_encoder,
            self.h264_hwaccel,
            self.h265_encoder,
            self.h265_hwaccel,
        )

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info:
            return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plan = build_compress_output_plan(self.config, out_dir, file_path, info)

        if not plan.has_work:
            return True

        cmd = self.build_ffmpeg_split_command(file_path, plan.active_maps, info)

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
            print(f"\n指令輸出結尾：\n{''.join(stderr_log)}")
            cleanup_temp_outputs(plan.tmps)
            return False

        tqdm_write(f"({idx}/{total}) {file_path.name} 處理完成!")
        promote_temp_outputs(plan.tmps, plan.finals)

        return True

    def run(self):
        run_video_batch(self.config, self.process_single_video, "進行自動壓縮")


def main():
    print("=" * 60)
    print("  Auto Compress Video - H.264 / H.265 自動解析度壓縮工具")
    print("=" * 60)

    config = CompressConfig()

    # --- 確認目標參數 ---
    print("\n【目標參數】")
    print(f"  輸出目標 (targets):")
    for i, t in enumerate(config.targets, 1):
        print(f"    [{i}] resolution={t['resolution']}  vcodec={t['vcodec']}")

    print()
    try:
        input("確認以上設定無誤後，請按 Enter 開始執行...")
    except EOFError:
        pass
    print()

    app = VideoCompressor(config)
    app.run()

    if should_pause_with_windows_prompt():
        pause_for_windows_shell()
    else:
        input("\n請按 Enter 鍵結束...")


if __name__ == "__main__":
    main()
