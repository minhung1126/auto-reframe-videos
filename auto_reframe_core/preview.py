"""Cancellable latest-request single-frame previews using production graphs."""

from dataclasses import replace
from pathlib import Path
import math
from threading import Condition, Thread
import tempfile

from auto_reframe_core.job_models import TaskContext
from auto_reframe_core.process_runner import ProcessRunner, TaskCancelled
from auto_reframe_core.platform_profile import hidden_subprocess_kwargs


def render_preview(mode, config, source, seconds, target_index=0, *, task_context=None):
    from auto_reframe_core.compress import VideoCompressor
    from auto_reframe_core.reframe import VideoReframer
    from auto_reframe_core.output_plans import (
        build_compress_output_plan,
        build_reframe_output_plan,
    )
    from auto_reframe_core.runtime_paths import resource_root, tool_path
    from auto_reframe_core.video_utils import get_video_info

    context = task_context or TaskContext()
    context.check()
    info = get_video_info(
        tool_path("ffprobe", config.ffprobe_path), source, task_context=context
    )
    if not info:
        raise ValueError("無法讀取預覽影片。")
    seconds = float(seconds)
    if (
        seconds < 0
        or not math.isfinite(seconds)
        or (info["duration"] and seconds >= info["duration"])
    ):
        raise ValueError("預覽時間必須位於影片範圍內。")
    config = replace(
        config,
        targets=[dict(config.targets[target_index])],
        skip_existing=False,
        prepared_batch=None,
    )
    processor_type = VideoReframer if mode == "reframe" else VideoCompressor
    processor = object.__new__(processor_type)
    processor.config, processor.task_context = config, context
    processor.script_dir = resource_root()
    processor.h264_encoder, processor.h265_encoder = "libx264", "libx265"
    processor.h264_hwaccel = processor.h265_hwaccel = None
    config.ffmpeg_path = tool_path("ffmpeg", config.ffmpeg_path)
    processor._validate_config()
    if mode == "reframe":
        processor.load_texts()
    with tempfile.TemporaryDirectory(prefix="arv-preview-") as temporary:
        root = Path(temporary)
        if mode == "reframe":
            target = config.targets[0]
            ratio = target["ratio"]
            dims = processor.calculate_dimensions(info["width"], info["height"], ratio)
            plan = build_reframe_output_plan(
                config, root, source, info, dims, *ratio, config.targets, prepare=False
            )
            command = processor.build_ffmpeg_split_command(
                source, dims, plan.active_maps, info, force_software_encode=True
            )
        else:
            plan = build_compress_output_plan(config, root, source, info, prepare=False)
            command = processor.build_ffmpeg_split_command(
                source, plan.active_maps, info, force_software_encode=True
            )
        prefix = (
            command[: command.index("-filter_complex") + 2]
            if "-filter_complex" in command
            else command[: command.index("-map")]
        )
        input_index = prefix.index("-i")
        prefix[input_index:input_index] = ["-ss", str(seconds)]
        video_map = command[command.index("-map") + 1]
        image = root / "preview.png"
        prefix += [
            "-map",
            video_map,
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-threads",
            "1",
            "-update",
            "1",
            str(image),
        ]
        result = ProcessRunner(context).capture(
            prefix, timeout=45, **hidden_subprocess_kwargs()
        )
        context.check()
        if result.returncode or not image.is_file():
            raise RuntimeError(result.stderr[-3000:] or "無法產生預覽畫面。")
        return image.read_bytes(), (
            plan.active_maps[0].width,
            plan.active_maps[0].height,
        )


class PreviewService:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.condition = Condition()
        self.pending = None
        self.context = None
        self.closed = False
        self.sequence = 0
        self.worker = Thread(target=self._run, name="video-preview", daemon=True)
        self.worker.start()

    @property
    def busy(self):
        with self.condition:
            return self.context is not None or self.pending is not None

    def request(self, *args):
        with self.condition:
            self.sequence += 1
            if self.context is not None:
                self.context.cancellation.request()
            context = TaskContext(event_sink=self.event_sink)
            self.pending = (context, args)
            self.condition.notify()
            return context.job_id

    def cancel(self, *, close=False):
        with self.condition:
            self.pending = None
            self.closed = self.closed or close
            if self.context is not None:
                self.context.cancellation.request()
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                context, args = self.pending
                self.pending, self.context = None, context
            try:
                payload = render_preview(*args, task_context=context)
                context.check()
                context.emit("preview_done", payload)
            except TaskCancelled:
                pass
            except Exception as error:
                if not context.cancellation.cancelled:
                    context.emit("preview_error", str(error))
            finally:
                with self.condition:
                    self.context = None
