# -*- coding: utf-8 -*-
"""Shared input scanning and parallel batch execution."""

from pathlib import Path
from typing import Callable, Tuple

from auto_reframe_core.video_utils import cleanup_tmp_files, resolve_workers, run_parallel


def run_video_batch(config, process_single_video: Callable, action_label: str) -> Tuple[int, list]:
    in_dir = Path(config.input_dir)
    input_was_missing = not in_dir.exists()
    in_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_was_missing:
        print(f"\n[提示] 未找到 '{in_dir.resolve()}'，已自動創建，請放置影片後重新執行。")
        return 0, []

    cleanup_tmp_files(out_dir)

    videos = [
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in config.video_extensions
    ]
    if not videos:
        print("\n[提示] 資料夾內無可支援的影片檔。")
        return 0, []

    workers = resolve_workers(config.max_workers)
    print(f"\n找到 {len(videos)} 個目標將開始{action_label} (平行任務數: {workers})...\n")

    tasks = [(i, len(videos), v) for i, v in enumerate(sorted(videos), 1)]
    try:
        success_count, failed_files = run_parallel(tasks, process_single_video, workers)
    except KeyboardInterrupt:
        print("\n[中斷] 任務已停止。已完成的輸出會保留，未完成的 .tmp 會在下次執行時清理。")
        return 0, []

    print("\n" + "=" * 60)
    print("  任務總結")
    print(f"  成功: {success_count} / {len(videos)}")
    if failed_files:
        print(f"  失敗: {', '.join(failed_files)}")
    print("=" * 60)

    return success_count, failed_files
