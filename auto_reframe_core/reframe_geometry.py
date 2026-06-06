# -*- coding: utf-8 -*-
"""Geometry calculations for fixed-canvas reframe output."""


def calculate_reframe_dimensions(src_w, src_h, target_ratio, final_ratio):
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

    f_w, f_h = final_ratio
    final_w, final_h = crop_w, int(crop_w * f_h / f_w)
    final_w += final_w % 2
    final_h += final_h % 2

    pad_top = max(0, (final_h - crop_h) // 2)
    pad_bottom = max(0, (final_h - crop_h) - pad_top)

    return {
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "final_w": final_w,
        "final_h": final_h,
    }
