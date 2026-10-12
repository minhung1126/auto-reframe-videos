# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0 - H.265)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：H.265 (HEVC) 高效壓縮、平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
"""

import sys
from pathlib import Path
from auto_reframe_core.runtime_paths import resource_root, tool_path, logs_root
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from queue import Queue

from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.ffmpeg_graphs import build_reframe_split_command
from auto_reframe_core.ffmpeg_execution import build_ffmpeg_attempts, execute_output_plan
from auto_reframe_core.encoder_profiles import detect_target_encoders
from auto_reframe_core.output_plans import (
    output_root_for,
    build_reframe_output_plan,
)
from auto_reframe_core.platform_profile import pause_for_windows_shell
from auto_reframe_core.reframe_geometry import calculate_reframe_dimensions
from auto_reframe_core.video_utils import (
    # 影片資訊
    get_video_info,
    # 共用常數
    h264, h265,
    # 執行工具
    emit_video_progress, run_ffmpeg_with_progress,
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
    video_files: tuple | None = None
    output_mode: str = "specified"
    prepared_batch: object | None = field(default=None, repr=False, compare=False)

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
    watermark_width_ratio: float = 0.15
    watermark_opacity: float = 0.8
    watermark_margin: int = 3

    # GUI 可直接提供文字；None 時維持既有文字檔讀取行為。
    top_text_override: Optional[str] = None
    bottom_text_override: Optional[str] = None

class VideoReframer:
    def __init__(self, config: ReframeConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.script_dir = resource_root()
        self.config.ffmpeg_path = tool_path("ffmpeg", self.config.ffmpeg_path)
        self.config.ffprobe_path = tool_path("ffprobe", self.config.ffprobe_path)
        self._validate_config()
        self.load_texts()
        profiles = detect_target_encoders(self.config.ffmpeg_path, {t["vcodec"] for t in self.config.targets})
        self.h265_encoder, self.h265_hwaccel = profiles.get(h265, ("libx265", None))
        self.h264_encoder, self.h264_hwaccel = profiles.get(h264, ("libx264", None))

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
        text_path = Path(self.config.input_dir).resolve().parent / filepath
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
        return build_ffmpeg_attempts(
            lambda **kwargs: self.build_ffmpeg_split_command(input_file, dims, active_maps, info, **kwargs),
            active_maps, self.h264_encoder, self.h265_encoder,
        )

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        cancellation = getattr(position_q, "cancellation", None)
        if cancellation is not None and cancellation.cancelled:
            return False
        prepared = getattr(self.config, "prepared_batch", None)
        info = (dict(prepared.validate_video(file_path).info) if prepared is not None
                else get_video_info(self.config.ffprobe_path, file_path))
        if not info:
            return False
        if cancellation is not None and cancellation.cancelled:
            return False

        out_dir = output_root_for(self.config, file_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        ratio_groups = {}
        for t in self.config.targets:
            ratio_groups.setdefault(t['ratio'], []).append(t)

        ratio_items = list(ratio_groups.items())
        group_total = len(ratio_items)
        all_success = True
        for group_index, ((rt_w, rt_h), targets) in enumerate(ratio_items, 1):
            if cancellation is not None and cancellation.cancelled:
                return False
            phase = f"比例 {rt_w}:{rt_h}（{group_index}/{group_total}）"
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))
            plan = (prepared.output_plan(file_path, ratio=(rt_w, rt_h), skip_existing=self.config.skip_existing)
                    if prepared is not None else build_reframe_output_plan(
                        self.config, out_dir, file_path, info, dims, rt_w, rt_h, targets))

            if not plan.has_work:
                emit_video_progress(
                    getattr(self, "progress_callback", None),
                    task_info,
                    kind="progress",
                    progress=100,
                    phase=f"{phase}－無需處理",
                )
                continue

            debug_log_path = None
            if self.config.debug:
                debug_log_path = logs_root() / f"ffmpeg_debug_{file_path.stem}_{rt_w}x{rt_h}.log"

            attempts = self._ffmpeg_attempts(file_path, dims, plan.active_maps, info)
            success = execute_output_plan(
                plan, attempts, info, task_info, position_q, runner=run_ffmpeg_with_progress,
                phase=phase, description=f"({idx}/{total}) {file_path.stem[:12]} [{rt_w}:{rt_h}]",
                progress_callback=getattr(self, "progress_callback", None), debug_log_path=debug_log_path,
                before_promotion=(lambda: prepared.validate_video(file_path)) if prepared is not None else None,
            )
            all_success = all_success and success

        return all_success

    def run(self):
        return run_video_batch(
            self.config,
            self.process_single_video,
            "轉換",
            progress_callback=getattr(self, "progress_callback", None),
        )


def main(input_dir="input", output_dir="output"):
    print("=" * 60)
    print("  Auto Reframe Video v2.0 - H.265 高效能優化版")
    print("=" * 60)

    config = ReframeConfig(input_dir=input_dir, output_dir=output_dir)
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
