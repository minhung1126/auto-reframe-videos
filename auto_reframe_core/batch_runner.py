"""GUI/CLI batch service with snapshots, output ownership and complete results."""
from dataclasses import replace
from pathlib import Path

from auto_reframe_core.job_models import (
    BatchResult, JobRequest, ResultStatus, TaskContext, VideoResult,
)
from auto_reframe_core.output_locks import OutputLocks
from auto_reframe_core.process_runner import TaskCancelled
from auto_reframe_core.video_utils import resolve_workers, run_parallel


def _video_snapshot(config, context):
    if getattr(config, "video_files", None) is not None:
        return tuple(Path(path).resolve() for path in config.video_files)
    root = Path(config.input_dir)
    if not root.is_dir():
        raise ValueError(f"輸入資料夾不存在：{root}")
    from auto_reframe_core.video_list import collect_videos
    videos, errors = collect_videos((root,), task_context=context)
    if errors and not videos:
        raise ValueError("\n".join(errors))
    return videos


def run_video_batch_result(config, process_single_video, action_label, progress_callback=None, *, task_context=None):
    from auto_reframe_core.output_plans import (
        prepare_video_batch, validate_prepared_batch, cleanup_temp_outputs, check_batch_space,
    )
    context = task_context or TaskContext()
    videos = ()
    try:
        context.check()
        videos = _video_snapshot(config, context)
        if not videos:
            return BatchResult(context.job_id)
        if not hasattr(config, "targets"):
            tasks = [(index, len(videos), video) for index, video in enumerate(videos, 1)]
            return run_parallel(tasks, process_single_video, resolve_workers(config.max_workers),
                                progress_callback=progress_callback, task_context=context, detailed=True)
        mode = "reframe" if hasattr(config, "final_ratio") else "compress"
        batch_config = replace(config, video_files=videos)
        if config.prepared_batch is None:
            config.prepared_batch = prepare_video_batch(batch_config, mode, task_context=context)
        prepared = config.prepared_batch
        batch_config.prepared_batch = prepared
        context.planned_targets = {source: tuple(path for _, plan in entry.plans for path in plan.finals)
                                   for source, entry in prepared.entries.items()}
        if context.output_lease is None:
            context.output_lease = OutputLocks(prepared.finals).acquire()
        validate_prepared_batch(batch_config, mode, task_context=context)
        check_batch_space(prepared, skip_existing=config.skip_existing)
        context.check()
        # Kernel locks are held before touching any fixed FFmpeg temp path.
        cleanup_temp_outputs([path.with_name(path.name + ".tmp") for path in prepared.finals])
        tasks = [(index, len(videos), video) for index, video in enumerate(videos, 1)]
        if getattr(process_single_video, "__self__", None) is not None:
            def process(task, positions):
                return process_single_video(task, positions, task_context=context)
        else:
            process = process_single_video
        result = run_parallel(tasks, process, resolve_workers(config.max_workers),
                              progress_callback=progress_callback, task_context=context, detailed=True)
        # Preserve injectable legacy parallel runners used by API callers.
        if not isinstance(result, BatchResult):
            successes, failures = result
            result = BatchResult(context.job_id, tuple(VideoResult(path, ResultStatus.FAILED if path.name in failures
                                  else ResultStatus.SUCCESS) for path in videos))
        return result
    except (TaskCancelled, KeyboardInterrupt):
        context.cancellation.cancel()
        return BatchResult(context.job_id, tuple(VideoResult(path, ResultStatus.CANCELLED) for path in videos))
    finally:
        context.release_outputs()


def run_video_batch(config, process_single_video, action_label, progress_callback=None):
    result = run_video_batch_result(config, process_single_video, action_label, progress_callback,
                                   task_context=getattr(getattr(process_single_video, "__self__", None), "task_context", None))
    print(f"\n任務總結：完成 {result.count(ResultStatus.SUCCESS)}；略過 {result.count(ResultStatus.SKIPPED)}；"
          f"失敗 {len(result.failed_sources)}；取消 {result.count(ResultStatus.CANCELLED)}")
    return result.legacy()


class BatchService:
    def prepare(self, request: JobRequest, context: TaskContext):
        from auto_reframe_core.output_plans import prepare_video_batch
        context.check()
        request.config.prepared_batch = prepare_video_batch(request.config, request.mode, task_context=context)
        return request.config.prepared_batch

    def reserve(self, request, context, *, delete_conflicts=()):
        from auto_reframe_core.output_plans import validate_prepared_batch, delete_planned_conflicts, check_batch_space
        prepared = request.config.prepared_batch
        context.check()
        context.output_lease = OutputLocks(prepared.finals).acquire()
        try:
            validate_prepared_batch(request.config, request.mode, task_context=context)
            check_batch_space(prepared, skip_existing=request.config.skip_existing)
            context.check()
            if delete_conflicts:
                delete_planned_conflicts(delete_conflicts, prepared.finals)
        except BaseException:
            context.release_outputs()
            raise

    def execute(self, request, context):
        from auto_reframe_core.reframe import VideoReframer
        from auto_reframe_core.compress import VideoCompressor
        processor_type = VideoReframer if request.mode == "reframe" else VideoCompressor
        try:
            context.check()
            processor = processor_type(request.config, progress_callback=lambda event: context.emit("video_progress", event),
                                       task_context=context)
            context.emit("encoders", {target["vcodec"]: getattr(processor, target["vcodec"] + "_encoder")
                                      for target in request.config.targets})
            return processor.run_result(context)
        finally:
            context.release_outputs()
