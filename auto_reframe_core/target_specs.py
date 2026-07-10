# -*- coding: utf-8 -*-
"""Shared target validation for reframe and compress configurations."""

from video_utils import RESOLUTION_MAP, h264, h265


VALID_CODECS = {h264, h265}


def _normalize_resolution_and_codec(target: dict, idx: int, config_name: str) -> None:
    resolution = str(target.get("resolution", "")).lower()
    vcodec = str(target.get("vcodec", "")).lower()

    if resolution not in RESOLUTION_MAP:
        allowed = ", ".join(sorted(RESOLUTION_MAP.keys()))
        raise ValueError(
            f"{config_name}.targets[{idx}].resolution 無效: {resolution!r}。可用值: {allowed}"
        )
    if vcodec not in VALID_CODECS:
        allowed = ", ".join(sorted(VALID_CODECS))
        raise ValueError(
            f"{config_name}.targets[{idx}].vcodec 無效: {vcodec!r}。可用值: {allowed}"
        )

    target["resolution"] = resolution
    target["vcodec"] = vcodec


def normalize_compress_target(target: dict, idx: int) -> None:
    config_name = "CompressConfig"
    if not isinstance(target, dict):
        raise ValueError(f"{config_name}.targets[{idx}] 必須是 dict。")
    _normalize_resolution_and_codec(target, idx, config_name)


def normalize_reframe_target(target: dict, idx: int, final_ratio) -> None:
    config_name = "ReframeConfig"
    if not isinstance(target, dict):
        raise ValueError(f"{config_name}.targets[{idx}] 必須是 dict。")

    ratio = target.get("ratio")
    if (
        not isinstance(ratio, (tuple, list))
        or len(ratio) != 2
        or int(ratio[0]) <= 0
        or int(ratio[1]) <= 0
    ):
        raise ValueError(
            f"{config_name}.targets[{idx}].ratio 必須是 2 個大於 0 的整數，例如 (4, 5)。"
        )

    rt_w, rt_h = int(ratio[0]), int(ratio[1])
    final_w, final_h = final_ratio
    if rt_w / rt_h < final_w / final_h:
        raise ValueError(
            f"{config_name}.targets[{idx}].ratio ({rt_w}:{rt_h}) 比 final_ratio ({final_w}:{final_h}) 更窄高，"
            "這將導致輸出影片比例失真變形，且不具備垂直補邊。請重新調整裁切比例。"
        )

    target["ratio"] = (rt_w, rt_h)
    _normalize_resolution_and_codec(target, idx, config_name)
