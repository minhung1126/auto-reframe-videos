# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0 - H.265)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：H.265 (HEVC) 高效壓縮、平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from queue import Queue

from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.ffmpeg_graphs import build_reframe_split_command
from auto_reframe_core.output_plans import (
    build_reframe_output_plan,
    cleanup_temp_outputs,
    promote_temp_outputs,
)
from auto_reframe_core.platform_profile import pause_for_windows_shell
from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
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
from auto_reframe_core.target_specs import normalize_reframe_target
from auto_reframe_core.text_layout import (
    TextLayoutConfig,
    escape_drawtext_text,
    escape_filter_path,
)

# 強制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class ReframeConfig:
    # --- 輸入 / 輸出 ---
    # 輸入影片的資料夾路徑
    input_dir: str = "input"
    # 輸出影片的資料夾路徑
    output_dir: str = "output"

    # --- 裁切與輸出目標 (重要設定) ---
    # resolution: '4k', '2k', '1080p', 'fhd', '720p', 'hd', '480p', '360p', 'source'
    # vcodec: h264, h265
    targets: List[dict] = field(default_factory=lambda: [
        {'ratio': (4, 5), 'resolution': 'source', 'vcodec': h265},
        {'ratio': (4, 5), 'resolution': '1080p',  'vcodec': h264},
    ])

    # 最終輸出的影片畫布比例 (預設為 9:16 直式)
    final_ratio: Tuple[int, int] = (9, 16)

    # --- 字幕與文字疊加 ---
    top_text_file: str = "top_text.txt"
    top_font_size: int = 48
    bottom_text_file: str = "bottom_text.txt"
    bottom_font_size: int = 24

    # --- 字型與樣式 ---
    font_path: str = "fonts/NotoSerifTC.ttf"
    font_color: str = "white"
    text_margin: int = 20
    top_text_line_spacing_ratio: float = 1.08
    bottom_text_line_spacing_ratio: float = 1.2

    # --- 系統與平行化設定 ---
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    video_extensions: set = field(default_factory=lambda: {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"
    })
    skip_existing: bool = True
    max_workers: int = 0
    debug: bool = False

    # --- 執行階段產生的內部變數 ---
    top_text_content: str = ""
    bottom_text_content: str = ""

class VideoReframer:
    def __init__(self, config: ReframeConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parent
        self._validate_config()
        self.load_texts()
        self.h265_encoder, self.h265_hwaccel = detect_h265_hw_encoder(self.config.ffmpeg_path)
        self.h264_encoder, self.h264_hwaccel = detect_h264_hw_encoder(self.config.ffmpeg_path)

    def _validate_config(self):
        if not self.config.targets:
            raise ValueError("ReframeConfig.targets 不可為空。")

        if (
            not isinstance(self.config.final_ratio, (tuple, list))
            or len(self.config.final_ratio) != 2
            or int(self.config.final_ratio[0]) <= 0
            or int(self.config.final_ratio[1]) <= 0
        ):
            raise ValueError("ReframeConfig.final_ratio 必須是 2 個大於 0 的整數。")
        self.config.final_ratio = (int(self.config.final_ratio[0]), int(self.config.final_ratio[1]))

        for idx, target in enumerate(self.config.targets, 1):
            normalize_reframe_target(target, idx, self.config.final_ratio)

    def load_texts(self):
        """讀取上下方的文字檔內容"""
        self.config.top_text_content, top_new = self._load_text_from_file(self.config.top_text_file)
        self.config.bottom_text_content, bottom_new = self._load_text_from_file(self.config.bottom_text_file)


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
        # 移除 \r 避免 FFmpeg 渲染出無法解析的方塊符號，並移除結尾的換行
        text = text.replace("\r", "").rstrip("\n")
        return text, False

    def calculate_dimensions(self, src_w, src_h, target_ratio):
        return calculate_reframe_dimensions(src_w, src_h, target_ratio, self.config.final_ratio)

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        return escape_filter_path(path)

    @staticmethod
    def _escape_drawtext_text(text: str) -> str:
        return escape_drawtext_text(text)

    def _line_spacing_px(self, font_size: int, line_spacing_ratio: float) -> int:
        extra_spacing = line_spacing_ratio - 1.0
        return int(round(font_size * extra_spacing))

    def _text_layout_config(self) -> TextLayoutConfig:
        layout = TextLayoutConfig(
            font_path=self._escape_filter_path(self.script_dir / self.config.font_path),
            font_color=self.config.font_color,
            text_margin=self.config.text_margin,
            top_font_size=self.config.top_font_size,
            bottom_font_size=self.config.bottom_font_size,
            top_line_spacing_ratio=self.config.top_text_line_spacing_ratio,
            bottom_line_spacing_ratio=self.config.bottom_text_line_spacing_ratio,
            top_text=self.config.top_text_content,
            bottom_text=self.config.bottom_text_content,
        )
        return layout

    def build_ffmpeg_split_command(self, input_file, dims, resolutions_map, info):
        """利用 FFmpeg -filter_complex 實作單次解碼多路輸出"""
        return build_reframe_split_command(
            self.config.ffmpeg_path,
            input_file,
            dims,
            resolutions_map,
            info,
            self.h264_encoder,
            self.h264_hwaccel,
            self.h265_encoder,
            self.h265_hwaccel,
            self._text_layout_config(),
        )

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info:
            return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ratio_groups = {}
        for t in self.config.targets:
            ratio_groups.setdefault(t['ratio'], []).append(t)

        all_success = True
        for (rt_w, rt_h), targets in ratio_groups.items():
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))
            plan = build_reframe_output_plan(
                self.config, out_dir, file_path, info, dims, rt_w, rt_h, targets
            )

            if not plan.has_work:
                continue

            cmd = self.build_ffmpeg_split_command(file_path, dims, plan.active_maps, info)

            debug_log_path = None
            if self.config.debug:
                debug_log_path = self.script_dir / f"ffmpeg_debug_{file_path.stem}_{rt_w}x{rt_h}.log"

            desc = f"({idx}/{total}) {file_path.stem[:12]} [{rt_w}:{rt_h}]"
            returncode, stderr_log = run_ffmpeg_with_progress(
                cmd, info, desc, position_q, debug_log_path
            )

            if returncode != 0:
                tqdm_write(f" [失敗!] ({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}]")
                print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
                print(f"指令輸出結尾：\n{''.join(stderr_log)}")
                cleanup_temp_outputs(plan.tmps)
                all_success = False
                continue

            tqdm_write(f"({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}] 完成!")
            promote_temp_outputs(plan.tmps, plan.finals)

        return all_success

    def run(self):
        run_video_batch(self.config, self.process_single_video, "轉換")


def main():
    print("=" * 60)
    print("  Auto Reframe Video v2.0 - H.265 高效能優化版")
    print("=" * 60)

    config = ReframeConfig()
    app = VideoReframer(config)

    # --- 確認目標參數與文字內容 ---
    print("\n【目標參數】")
    print(f"  最終畫布比例 (final_ratio): {config.final_ratio[0]}:{config.final_ratio[1]}")
    print(f"  輸出目標 (targets):")
    for i, t in enumerate(config.targets, 1):
        print(f"    [{i}] ratio={t['ratio'][0]}:{t['ratio'][1]}  resolution={t['resolution']}  vcodec={t['vcodec']}")

    print("\n【文字內容】")
    top_lines = config.top_text_content.splitlines() if config.top_text_content else []
    btm_lines = config.bottom_text_content.splitlines() if config.bottom_text_content else []
    print(f"  ▲ top_text.txt ({len(top_lines)} 行):")
    if top_lines:
        for ln in top_lines:
            print(f"    | {ln}")
    else:
        print("    (空)")
    print(f"  ▼ bottom_text.txt ({len(btm_lines)} 行):")
    if btm_lines:
        for ln in btm_lines:
            print(f"    | {ln}")
    else:
        print("    (空)")

    print()
    try:
        input("確認以上設定無誤後，請按 Enter 開始執行...")
    except EOFError:
        pass
    print()

    app.run()

    pause_for_windows_shell()


if __name__ == "__main__":
    main()
