# -*- coding: utf-8 -*-
"""FFmpeg command builders for compress and fixed-layout reframe jobs."""

from collections import OrderedDict
from pathlib import Path
from typing import Callable

from .text_layout import TextLayoutConfig, append_fixed_reframe_text_filters
from .watermark import (
    WatermarkConfig,
    append_watermark_overlay_filter,
    append_watermark_source_filter,
)
from auto_reframe_core.video_utils import (
    build_output_args,
    detect_hwaccel_for_cmd,
    h264,
    h265,
)


def _collect_hwaccels(outputs: list, encoder_hwaccel_for_codec: Callable[[str], str]) -> set:
    hwaccels = set()
    for entry in outputs:
        vcodec = entry[5] if len(entry) > 5 else h265
        hw = encoder_hwaccel_for_codec(vcodec)
        if hw:
            hwaccels.add(hw)
    return hwaccels


def _encoder_for_codec(vcodec: str, h264_encoder: str, h265_encoder: str) -> str:
    if vcodec == h265:
        return h265_encoder
    if vcodec == h264:
        return h264_encoder
    raise ValueError(f"不支援的 vcodec: {vcodec!r}")


def build_reframe_split_command(
    ffmpeg_path: str,
    input_file: Path,
    dims: dict,
    outputs: list,
    info: dict,
    h264_encoder: str,
    h264_hwaccel: str,
    h265_encoder: str,
    h265_hwaccel: str,
    layout: TextLayoutConfig,
    watermark: WatermarkConfig = WatermarkConfig(),
) -> list:
    cmd = [ffmpeg_path, "-hide_banner", "-y"]
    cmd += detect_hwaccel_for_cmd(
        _collect_hwaccels(outputs, lambda codec: h265_hwaccel if codec == h265 else h264_hwaccel)
    )
    cmd += ["-i", str(input_file)]
    if watermark.enabled:
        cmd += ["-i", str(watermark.path)]

    filters = []
    crop = f"crop={dims['crop_w']}:{dims['crop_h']}:{dims['crop_x']}:{dims['crop_y']}"
    pad = (
        f"pad={dims['final_w']}:{dims['final_h']}:0:{dims['pad_top']}:black"
        if dims["pad_top"] > 0 or dims["pad_bottom"] > 0 else ""
    )

    base = f"[0:v]{crop}[crp]"
    if pad:
        base += f";[crp]{pad}[pad]"
        root_label = "[pad]"
    else:
        root_label = "[crp]"

    res_groups = OrderedDict()
    for index, entry in enumerate(outputs):
        key = (entry[0], entry[1])
        res_groups.setdefault(key, []).append(index)

    split_count = len(res_groups)
    if split_count > 1:
        split_labels = "".join([f"[base_{i}]" for i in range(split_count)])
        base += f";{root_label}split={split_count}{split_labels}"
        base_inputs = [f"[base_{i}]" for i in range(split_count)]
    else:
        base_inputs = [root_label]

    filters.append(base)
    watermark_inputs = append_watermark_source_filter(
        filters, split_count, watermark
    )

    final_video_maps = [None] * len(outputs)

    for group_index, ((out_w, out_h), indices) in enumerate(res_groups.items()):
        res_label = f"[res_{group_index}]"
        seq = f"{base_inputs[group_index]}scale={out_w}:{out_h}:flags=bicubic{res_label}"
        curr_label = res_label

        seq, curr_label = append_fixed_reframe_text_filters(
            seq, curr_label, group_index, out_h, dims, layout
        )
        filters.append(seq)

        if watermark.enabled:
            curr_label = append_watermark_overlay_filter(
                filters,
                curr_label,
                watermark_inputs[group_index],
                group_index,
                out_w,
                out_h,
                watermark,
            )

        num_codecs = len(indices)
        if num_codecs > 1:
            codec_split_labels = "".join([f"[out_{idx}]" for idx in indices])
            filters.append(f"{curr_label}split={num_codecs}{codec_split_labels}")
            for idx in indices:
                final_video_maps[idx] = f"[out_{idx}]"
        else:
            final_video_maps[indices[0]] = curr_label

    cmd += ["-filter_complex", ";".join(filters)]

    for index, entry in enumerate(outputs):
        _, _, _, bitrate, out_file = entry[:5]
        vcodec = entry[5] if len(entry) > 5 else h265
        cmd += ["-map", final_video_maps[index]]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0"]
        encoder = _encoder_for_codec(vcodec, h264_encoder, h265_encoder)
        cmd += build_output_args(encoder, bitrate, vcodec, info["has_audio"], out_file)

    return cmd


def build_compress_split_command(
    ffmpeg_path: str,
    input_file: Path,
    outputs: list,
    info: dict,
    h264_encoder: str,
    h264_hwaccel: str,
    h265_encoder: str,
    h265_hwaccel: str,
    watermark: WatermarkConfig = WatermarkConfig(),
) -> list:
    cmd = [ffmpeg_path, "-hide_banner", "-y"]
    cmd += detect_hwaccel_for_cmd(
        _collect_hwaccels(outputs, lambda codec: h265_hwaccel if codec == h265 else h264_hwaccel)
    )
    cmd += ["-i", str(input_file)]
    if watermark.enabled:
        cmd += ["-i", str(watermark.path)]

    res_groups = OrderedDict()
    for index, entry in enumerate(outputs):
        key = (entry[0], entry[1])
        res_groups.setdefault(key, []).append(index)

    num_resolutions = len(res_groups)
    filters = []
    final_video_maps = [None] * len(outputs)

    if num_resolutions > 1:
        split_labels = "".join([f"[raw_{j}]" for j in range(num_resolutions)])
        filters.append(f"[0:v]split={num_resolutions}{split_labels}")
        raw_inputs = [f"[raw_{j}]" for j in range(num_resolutions)]
    else:
        raw_inputs = ["[0:v]"]

    watermark_inputs = append_watermark_source_filter(
        filters, num_resolutions, watermark
    )

    for group_index, ((out_w, out_h), indices) in enumerate(res_groups.items()):
        num_codecs = len(indices)
        if out_w == info["width"] and out_h == info["height"]:
            curr_input = raw_inputs[group_index]
        else:
            scaled_label = f"[scaled_{group_index}]"
            filters.append(f"{raw_inputs[group_index]}scale={out_w}:{out_h}:flags=bicubic{scaled_label}")
            curr_input = scaled_label

        if watermark.enabled:
            curr_input = append_watermark_overlay_filter(
                filters,
                curr_input,
                watermark_inputs[group_index],
                group_index,
                out_w,
                out_h,
                watermark,
            )

        if num_codecs > 1:
            codec_split_labels = "".join([f"[out_{idx}]" for idx in indices])
            filters.append(f"{curr_input}split={num_codecs}{codec_split_labels}")
            for idx in indices:
                final_video_maps[idx] = f"[out_{idx}]"
        else:
            final_video_maps[indices[0]] = curr_input

    if filters:
        cmd += ["-filter_complex", ";".join(filters)]

    for index, (_, _, _, bitrate, out_file, vcodec) in enumerate(outputs):
        video_map = final_video_maps[index]
        if not filters and video_map.startswith("[") and video_map.endswith("]"):
            video_map = video_map[1:-1]

        cmd += ["-map", video_map]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0"]
        encoder = _encoder_for_codec(vcodec, h264_encoder, h265_encoder)
        cmd += build_output_args(encoder, bitrate, vcodec, info["has_audio"], out_file)

    return cmd
