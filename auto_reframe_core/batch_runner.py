# -*- coding: utf-8 -*-
"""Shared input scanning and parallel batch execution."""

from pathlib import Path
from typing import Callable, Tuple

from auto_reframe_core.video_utils import resolve_workers, run_parallel


def run_video_batch(
    config,
    process_single_video: Callable,
    action_label: str,
    progress_callback=None,
) -> Tuple[int, list]:
    if getattr(config, "video_files", None) is not None:
        videos = tuple(Path(path) for path in config.video_files)
    else:
        in_dir = Path(config.input_dir)
        if not in_dir.is_dir():
            raise ValueError(f"輸入資料夾不存在：{in_dir}")
        videos = tuple(sorted(f for f in in_dir.iterdir()
                              if f.is_file() and f.suffix.lower() in config.video_extensions))
    from auto_reframe_core.output_plans import output_root_for, preflight_outputs, cleanup_temp_outputs
    if videos and hasattr(config, "targets"):
        from dataclasses import replace
        _, finals = preflight_outputs(replace(config, video_files=videos),
                          "reframe" if hasattr(config, "final_ratio") else "compress")
        cleanup_temp_outputs([path.with_name(path.name + ".tmp") for path in finals])
    for out_dir in {output_root_for(config, path) for path in videos}:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Each processor cleans only its own temporary outputs on failure/cancellation.

    if not videos:
        print("\n[提示] 資料夾內無可支援的影片檔。")
        return 0, []

    workers = resolve_workers(config.max_workers)
    print(f"\n找到 {len(videos)} 個目標將開始{action_label} (平行任務數: {workers})...\n")

    tasks = [(i, len(videos), v) for i, v in enumerate(videos, 1)]
    try:
        success_count, failed_files = run_parallel(
            tasks,
            process_single_video,
            workers,
            progress_callback=progress_callback,
        )
    except KeyboardInterrupt:
        print("\n[中斷] 任務已停止。已完成的輸出會保留，本次未完成的 .tmp 由 worker 清理。")
        return 0, []

    print("\n" + "=" * 60)
    print("  任務總結")
    print(f"  成功: {success_count} / {len(videos)}")
    if failed_files:
        print(f"  失敗: {', '.join(failed_files)}")
    print("=" * 60)

    return success_count, failed_files
