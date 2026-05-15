# -*- coding: utf-8 -*-
"""
Auto Reframe Video — 橫轉直影片工具 (v2.0 - H.265)
將橫向影片透過 ffmpeg 轉為手機直式影片 (9:16)
優化項目：H.265 (HEVC) 高效壓縮、平行處理、單次解碼多路輸出、避免重複讀檔、即時運算進度條、物件導向重構
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
    get_youtube_bitrate,
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

VALID_CODECS = {h264, h265}

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
            if not isinstance(target, dict):
                raise ValueError(f"ReframeConfig.targets[{idx}] 必須是 dict。")

            ratio = target.get("ratio")
            if (
                not isinstance(ratio, (tuple, list))
                or len(ratio) != 2
                or int(ratio[0]) <= 0
                or int(ratio[1]) <= 0
            ):
                raise ValueError(
                    f"ReframeConfig.targets[{idx}].ratio 必須是 2 個大於 0 的整數，例如 (4, 5)。"
                )
            rt_w, rt_h = int(ratio[0]), int(ratio[1])

            resolution = str(target.get("resolution", "")).lower()
            vcodec = str(target.get("vcodec", "")).lower()

            if resolution not in RESOLUTION_MAP:
                allowed = ", ".join(sorted(RESOLUTION_MAP.keys()))
                raise ValueError(
                    f"ReframeConfig.targets[{idx}].resolution 無效: {resolution!r}。可用值: {allowed}"
                )
            if vcodec not in VALID_CODECS:
                allowed = ", ".join(sorted(VALID_CODECS))
                raise ValueError(
                    f"ReframeConfig.targets[{idx}].vcodec 無效: {vcodec!r}。可用值: {allowed}"
                )

            target["ratio"] = (rt_w, rt_h)
            target["resolution"] = resolution
            target["vcodec"] = vcodec

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
        t_w, t_h = target_ratio
        source_ratio = src_w / src_h
        target_ratio_val = t_w / t_h

        if source_ratio > target_ratio_val:
            crop_h = src_h
            crop_w = int(src_h * target_ratio_val)
        else:
            crop_w = src_w
            crop_h = int(src_w / target_ratio_val)

        crop_w = min(crop_w - crop_w % 2, src_w)
        crop_h = min(crop_h - crop_h % 2, src_h)
        crop_x, crop_y = (src_w - crop_w) // 2, (src_h - crop_h) // 2

        f_w, f_h = self.config.final_ratio
        final_w, final_h = crop_w, int(crop_w * f_h / f_w)
        final_w += final_w % 2
        final_h += final_h % 2

        pad_top = max(0, (final_h - crop_h) // 2)
        pad_bottom = max(0, (final_h - crop_h) - pad_top)

        return {
            "crop_w": crop_w, "crop_h": crop_h, "crop_x": crop_x, "crop_y": crop_y,
            "pad_top": pad_top, "pad_bottom": pad_bottom, "final_w": final_w, "final_h": final_h,
        }

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        return str(path).replace("\\", "/").replace(":", "\\:")

    @staticmethod
    def _escape_drawtext_text(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace(":", "\\:")
            .replace("%", "%%")
        )

    def _line_spacing_px(self, font_size: int, line_spacing_ratio: float) -> int:
        extra_spacing = line_spacing_ratio - 1.0
        return int(round(font_size * extra_spacing))

    def build_ffmpeg_split_command(self, input_file, dims, resolutions_map, info):
        """利用 FFmpeg -filter_complex 實作單次解碼多路輸出"""
        cmd = [self.config.ffmpeg_path, "-hide_banner", "-y"]

        # 收集所有輸出的 hwaccel 後端，用共用工具決定是否啟用解碼端加速
        # (VideoToolbox 解碼端已透過 detect_hwaccel_for_cmd 排除，避免像素格式衝突)
        hwaccels = set()
        for entry in resolutions_map:
            vcodec = entry[5] if len(entry) > 5 else h265
            hw = self.h265_hwaccel if vcodec == h265 else self.h264_hwaccel
            if hw:
                hwaccels.add(hw)
        cmd += detect_hwaccel_for_cmd(hwaccels)

        cmd += ["-i", str(input_file)]

        filters = []
        crop = f"crop={dims['crop_w']}:{dims['crop_h']}:{dims['crop_x']}:{dims['crop_y']}"
        pad = (
            f"pad={dims['final_w']}:{dims['final_h']}:0:{dims['pad_top']}:black"
            if dims['pad_top'] > 0 or dims['pad_bottom'] > 0 else ""
        )

        base = f"[0:v]{crop}[crp]"
        if pad:
            base += f";[crp]{pad}[pad]"
            root_lbl = "[pad]"
        else:
            root_lbl = "[crp]"

        res_groups = OrderedDict()
        for i, entry in enumerate(resolutions_map):
            key = (entry[0], entry[1])
            res_groups.setdefault(key, []).append(i)

        splits_cnt = len(res_groups)
        if splits_cnt > 1:
            split_lbls = "".join([f"[base_{i}]" for i in range(splits_cnt)])
            base += f";{root_lbl}split={splits_cnt}{split_lbls}"
            base_inputs = [f"[base_{i}]" for i in range(splits_cnt)]
        else:
            base_inputs = [root_lbl]

        filters.append(base)

        font_path = self._escape_filter_path(self.script_dir / self.config.font_path)
        top_txt = self.config.top_text_content
        btm_txt = self.config.bottom_text_content

        final_video_maps = [None] * len(resolutions_map)

        for j, ((out_w, out_h), indices) in enumerate(res_groups.items()):
            res_lbl = f"[res_{j}]"
            seq = f"{base_inputs[j]}scale={out_w}:{out_h}:flags=bicubic{res_lbl}"
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
                    text_esc = self._escape_drawtext_text(ln)
                    rev_i = len(lines) - 1 - ln_i
                    y_pos = f"{ptop}-{mar}-text_h-{rev_i}*line_h*{self.config.top_text_line_spacing_ratio}"
                    next_lbl = f"[t_{j}_{ln_i}]"
                    seq += (f";{curr_lbl}drawtext=fontfile='{font_path}':text='{text_esc}':fontsize={fz}:"
                            f"fontcolor={self.config.font_color}:borderw={bw}:bordercolor={border_c}:"
                            f"x=(w-text_w)/2:y={y_pos}{next_lbl}")
                    curr_lbl = next_lbl

            if btm_txt:
                fz = int(self.config.bottom_font_size * scale_rate)
                bw = max(1, int(fz * 0.03))
                mar = int(self.config.text_margin * scale_rate)
                pbtm = int(dims["pad_bottom"] * (out_h / dims["final_h"]))
                lines = btm_txt.splitlines()
                for ln_i, ln in enumerate(lines):
                    text_esc = self._escape_drawtext_text(ln)
                    y_pos = f"{out_h}-{pbtm}+{mar}+{ln_i}*line_h*{self.config.bottom_text_line_spacing_ratio}"
                    next_lbl = f"[b_{j}_{ln_i}]"
                    seq += (f";{curr_lbl}drawtext=fontfile='{font_path}':text='{text_esc}':fontsize={fz}:"
                            f"fontcolor={self.config.font_color}:borderw={bw}:bordercolor={border_c}:"
                            f"x=(w-text_w)/2:y={y_pos}{next_lbl}")
                    curr_lbl = next_lbl

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
            vcodec = entry[5] if len(entry) > 5 else h265
            cmd += ["-map", final_video_maps[i]]
            if info["has_audio"]:
                cmd += ["-map", "0:a:0"]
            if vcodec == h265:
                encoder = self.h265_encoder
            elif vcodec == h264:
                encoder = self.h264_encoder
            else:
                raise ValueError(f"不支援的 vcodec: {vcodec!r}")
            cmd += build_output_args(encoder, vbr, vcodec, info["has_audio"], out_file)

        return cmd

    def process_single_video(self, task_info: Tuple[int, int, Path], position_q: Queue) -> bool:
        idx, total, file_path = task_info
        info = get_video_info(self.config.ffprobe_path, file_path)
        if not info:
            return False

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ratio_groups = OrderedDict()
        for t in self.config.targets:
            ratio_groups.setdefault(t['ratio'], []).append(t)

        all_success = True
        for (rt_w, rt_h), targets in ratio_groups.items():
            dims = self.calculate_dimensions(info["width"], info["height"], (rt_w, rt_h))

            active_maps = []
            tmps = []
            finals = []

            for t in targets:
                res_key = t['resolution'].lower()
                vcodec = t['vcodec']

                f_w, f_h = self.config.final_ratio
                source_short = min(dims['final_w'], dims['final_h'])
                final_short = resolve_short_side(res_key, source_short)

                if f_w <= f_h:
                    out_w = final_short
                    out_h = int(final_short * f_h / f_w)
                else:
                    out_h = final_short
                    out_w = int(final_short * f_w / f_h)

                out_w += out_w % 2
                out_h += out_h % 2

                suffix_name = f"{rt_w}x{rt_h}_{resolution_label(final_short)}_{vcodec}"
                sub_dir = out_dir / suffix_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                target_f = sub_dir / f"{file_path.stem}_{suffix_name}.mp4"

                if self.config.skip_existing and target_f.exists():
                    continue

                tmp_f = target_f.with_name(target_f.name + ".tmp")
                bitrate = get_youtube_bitrate(min(out_w, out_h), info["fps"])
                active_maps.append((out_w, out_h, resolution_label(final_short), bitrate, tmp_f, vcodec))
                tmps.append(tmp_f)
                finals.append(target_f)

            if not active_maps:
                continue

            cmd = self.build_ffmpeg_split_command(file_path, dims, active_maps, info)

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
                for t in tmps:
                    if t.exists():
                        t.unlink()
                all_success = False
                continue

            tqdm_write(f"({idx}/{total}) {file_path.name} [{rt_w}:{rt_h}] 完成!")

            for t, f in zip(tmps, finals):
                if t.exists():
                    if f.exists():
                        f.unlink()
                    t.rename(f)

        return all_success

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
        print(f"\n找到 {len(videos)} 個目標將開始轉換 (平行任務數: {workers})...\n")

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
    print("  Auto Reframe Video v2.0 - H.265 高效能優化版")
    print("=" * 60)

    config = ReframeConfig()
    app = VideoReframer(config)
    app.run()

    if os.name == 'nt':
        os.system("pause")


if __name__ == "__main__":
    main()
