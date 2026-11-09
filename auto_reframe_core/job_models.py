"""Tk-independent requests, ownership, events and per-target outcomes."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, NamedTuple
from uuid import uuid4

from auto_reframe_core.process_runner import ProcessCancellation


class OutputTarget(NamedTuple):
    width: int
    height: int
    label: str
    bitrate: str
    path: Path
    codec: str


class ResultStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass(frozen=True)
class JobRequest:
    mode: str
    config: Any
    job_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    kind: str
    payload: Any = None


@dataclass(frozen=True)
class TargetResult:
    path: Path
    status: ResultStatus
    message: str = ""
    error_kind: str = ""


@dataclass(frozen=True)
class VideoResult:
    source: Path
    status: ResultStatus
    targets: tuple[TargetResult, ...] = ()
    error_kind: str = ""
    message: str = ""


@dataclass(frozen=True)
class BatchResult:
    job_id: str
    videos: tuple[VideoResult, ...] = ()

    def count(self, status):
        return sum(video.status == status for video in self.videos)

    @property
    def completed_outputs(self):
        return sum(
            target.status == ResultStatus.SUCCESS
            for video in self.videos
            for target in video.targets
        )

    @property
    def status(self):
        if any(video.status == ResultStatus.CANCELLED for video in self.videos):
            return ResultStatus.CANCELLED
        if any(
            video.status in {ResultStatus.FAILED, ResultStatus.PARTIAL}
            for video in self.videos
        ):
            return (
                ResultStatus.PARTIAL
                if any(
                    video.status in {ResultStatus.SUCCESS, ResultStatus.SKIPPED}
                    for video in self.videos
                )
                else ResultStatus.FAILED
            )
        return ResultStatus.SUCCESS

    @property
    def failed_sources(self):
        return tuple(
            video.source
            for video in self.videos
            if video.status in {ResultStatus.FAILED, ResultStatus.PARTIAL}
        )

    def legacy(self):
        return (
            self.count(ResultStatus.SUCCESS) + self.count(ResultStatus.SKIPPED),
            [
                video.source.name
                for video in self.videos
                if video.status in {ResultStatus.FAILED, ResultStatus.PARTIAL}
            ],
        )


@dataclass
class TaskContext:
    job_id: str = field(default_factory=lambda: uuid4().hex)
    cancellation: ProcessCancellation = field(default_factory=ProcessCancellation)
    event_sink: Callable | None = None
    output_lease: Any = None
    target_results: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    planned_targets: dict = field(default_factory=dict)

    def check(self):
        self.cancellation.check()

    def emit(self, kind, payload=None):
        if kind in {"log", "encoders", "error", "done", "cancelled"}:
            from auto_reframe_core.diagnostics import local_logger

            try:
                # Event summaries keep full media/config data out of persistent logs.
                local_logger().info(
                    "job=%s event=%s %s",
                    self.job_id,
                    kind,
                    str(payload)[:3000] if kind in {"log", "encoders", "error"} else "",
                )
            except OSError:
                pass
        if self.event_sink is not None:
            self.event_sink(JobEvent(self.job_id, kind, payload))

    def record_targets(self, source, paths, status, message="", error_kind=""):
        entries = self.target_results.setdefault(Path(source), {})
        entries.update(
            {
                Path(path): TargetResult(
                    Path(path), ResultStatus(status), message, error_kind
                )
                for path in paths
            }
        )

    def release_outputs(self):
        if self.output_lease is not None:
            self.output_lease.release()
            self.output_lease = None
