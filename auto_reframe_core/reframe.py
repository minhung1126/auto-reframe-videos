# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0 - H.265)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：H.265 (HEVC) 高效壓縮、平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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
from auto_reframe_core.video_utils import (
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
from auto_reframe_core.watermark import (
    WatermarkConfig,
    build_watermark_config,
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

    # --- 圖片浮水印（相對路徑以專案目錄為基準） ---
    watermark_enabled: bool = False
    watermark_file: str = ""
    watermark_position: str = "bottom-center"
    watermark_width_ratio: float = 0.07
    watermark_opacity: float = 0.85
    watermark_margin: int = 3

    # GUI 可直接提供文字；None 時維持既有文字檔讀取行為。
    top_text_override: Optional[str] = None
    bottom_text_override: Optional[str] = None

class VideoReframer:
    def __init__(self, config: ReframeConfig):
        self.config = config
        self.script_dir = Path(__file__).resolve().parents[1]
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

        self.watermark = build_watermark_config(
            enabled=self.config.watermark_enabled,
            watermark_file=self.config.watermark_file,
            position=self.config.watermark_position,
            width_ratio=self.config.watermark_width_ratio,
            opacity=self.config.watermark_opacity,
            margin=self.config.watermark_margin,
            base_dir=self.script_dir,
        )

    def load_texts(self):
        """讀取上下方的文字檔內容"""
        if self.config.top_text_override is None:
            self.config.top_text_content, _ = self._load_text_from_file(
                self.config.top_text_file
            )
        else:
            self.config.top_text_content = self._normalize_text(
                self.config.top_text_override
            )

        if self.config.bottom_text_override is None:
            self.config.bottom_text_content, _ = self._load_text_from_file(
                self.config.bottom_text_file
            )
        else:
            self.config.bottom_text_content = self._normalize_text(
                self.config.bottom_text_override
            )


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
                try:
                    # 嘗試使用本機系統預設編碼，例如 Windows 的 CP950/CP936
                    import locale
                    text = text_path.read_text(encoding=locale.getpreferredencoding())
                    print(f"  [提示] 使用系統預設編碼讀取文字檔: {text_path}")
                except Exception:
                    print(f"  [警告] 無法讀取文字檔 (不支援的編碼): {text_path}")
                    return "", False
        return self._normalize_text(text), False

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 移除 \r 避免 FFmpeg 渲染出無法解析的方塊符號，並只移除結尾換行。
        return str(text).replace("\r", "").rstrip("\n")

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

    def build_ffmpeg_split_command(
        self,
        input_file,
        dims,
        resolutions_map,
        info,
        disable_decode_hwaccel: bool = False,
        force_software_encode: bool = False,
    ):
        """利用 FFmpeg -filter_complex 實作單次解碼多路輸出"""
        h264_encoder = "libx264" if force_software_encode else self.h264_encoder
        h265_encoder = "libx265" if force_software_encode else self.h265_encoder
        h264_hwaccel = None if disable_decode_hwaccel else self.h264_hwaccel
        h265_hwaccel = None if disable_decode_hwaccel else self.h265_hwaccel
        return build_reframe_split_command(
            self.config.ffmpeg_path,
            input_file,
            dims,
            resolutions_map,
            info,
            h264_encoder,
            h264_hwaccel,
            h265_encoder,
            h265_hwaccel,
            self._text_layout_config(),
            getattr(self, "watermark", WatermarkConfig()),
        )

    def _ffmpeg_attempts(self, input_file, dims, active_maps, info):
        first_cmd = self.build_ffmpeg_split_command(
            input_file, dims, active_maps, info
        )
        attempts = [("硬體優先", first_cmd)]
        if "-hwaccel" in first_cmd:
            attempts.append(
                (
                    "停用硬體解碼、保留硬體編碼",
                    self.build_ffmpeg_split_command(
                        input_file,
                        dims,
                        active_maps,
                        info,
                        disable_decode_hwaccel=True,
                    ),
                )
            )

        uses_hardware_encoder = any(
            (entry[5] == h264 and self.h264_encoder != "libx264")
            or (entry[5] == h265 and self.h265_encoder != "libx265")
            for entry in active_maps
        )
        if uses_hardware_encoder:
            attempts.append(
                (
                    "軟體解碼與軟體編碼",
                    self.build_ffmpeg_split_command(
                        input_file,
                        dims,
                        active_maps,
                        info,
                        disable_decode_hwaccel=True,
                        force_software_encode=True,
                    ),
                )
            )
        return attempts

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        cancellation = getattr(position_q, "cancellation", None)
        if cancellation is not None and cancellation.cancelled:
            return False
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info:
            return False
        if cancellation is not None and cancellation.cancelled:
            return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ratio_groups = {}
        for t in self.config.targets:
            ratio_groups.setdefault(t['ratio'], []).append(t)

        all_success = True
        for (rt_w, rt_h), targets in ratio_groups.items():
            if cancellation is not None and cancellation.cancelled:
                return False
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))
            plan = build_reframe_output_plan(
                self.config, out_dir, file_path, info, dims, rt_w, rt_h, targets
            )

            if not plan.has_work:
                continue

            debug_log_path = None
            if self.config.debug:
                debug_log_path = self.script_dir / f"ffmpeg_debug_{file_path.stem}_{rt_w}x{rt_h}.log"

            attempts = self._ffmpeg_attempts(file_path, dims, plan.active_maps, info)
            returncode = 1
            stderr_log = []
            for attempt_index, (attempt_label, cmd) in enumerate(attempts, 1):
                if cancellation is not None and cancellation.cancelled:
                    cleanup_temp_outputs(plan.tmps)
                    return False
                desc = f"({idx}/{total}) {file_path.stem[:12]} [{rt_w}:{rt_h}]"
                if attempt_index > 1:
                    tqdm_write(f"  [重試] {file_path.name}: {attempt_label}")
                    desc += f" [重試 {attempt_index}]"

                returncode, stderr_log = run_ffmpeg_with_progress(
                    cmd, info, desc, position_q, debug_log_path
                )
                if cancellation is not None and cancellation.cancelled:
                    cleanup_temp_outputs(plan.tmps)
                    return False
                if returncode == 0:
                    break
                cleanup_temp_outputs(plan.tmps)

            if returncode != 0:
                tqdm_write(f" [失敗!] ({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}]")
                print(f"\n\n[FFmpeg Error] 處理影片 {file_path.name} 時失敗！")
                print(f"指令輸出結尾：\n{''.join(stderr_log)}")
                all_success = False
                continue

            tqdm_write(f"({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}] 完成!")
            promote_temp_outputs(plan.tmps, plan.finals)

        return all_success

    def run(self):
        return run_video_batch(self.config, self.process_single_video, "轉換")


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
