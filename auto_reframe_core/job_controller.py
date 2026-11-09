"""Legal job transitions and bounded delivery, independent of any GUI toolkit."""

from collections import OrderedDict
from enum import Enum
from queue import Queue, Empty
from threading import Lock, Thread

from auto_reframe_core.job_models import JobEvent, TaskContext


class JobState(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    PREPARING = "preparing"
    CONFIRMING = "confirming"
    VALIDATING = "validating"
    RUNNING = "running"
    CANCELLING = "cancelling"
    UPDATING = "updating"
    FINISHED = "finished"
    CLOSED = "closed"


_TRANSITIONS = {
    JobState.IDLE: {
        JobState.SCANNING,
        JobState.PREPARING,
        JobState.UPDATING,
        JobState.CLOSED,
    },
    JobState.FINISHED: {
        JobState.SCANNING,
        JobState.PREPARING,
        JobState.UPDATING,
        JobState.CLOSED,
    },
    JobState.SCANNING: {JobState.CANCELLING, JobState.FINISHED},
    JobState.PREPARING: {JobState.CONFIRMING, JobState.CANCELLING, JobState.FINISHED},
    JobState.CONFIRMING: {JobState.VALIDATING, JobState.CANCELLING, JobState.FINISHED},
    JobState.VALIDATING: {JobState.RUNNING, JobState.CANCELLING, JobState.FINISHED},
    JobState.RUNNING: {JobState.CANCELLING, JobState.FINISHED},
    JobState.CANCELLING: {JobState.FINISHED},
    JobState.UPDATING: {JobState.FINISHED},
}


class EventBuffer:
    """Progress coalesces; bounded lifecycle backpressure never blocks cancellation."""

    def __init__(self, maxsize=2048, progress_limit=1024):
        self.control = Queue(maxsize=maxsize)
        self.progress = OrderedDict()
        self.progress_limit = progress_limit
        self.lock = Lock()

    def put(self, item):
        kind, payload = (
            (item.kind, item.payload) if isinstance(item, JobEvent) else item
        )
        if kind in {"video_progress", "update_progress", "log"}:
            event_kind = getattr(payload, "kind", "")
            if kind != "video_progress" or event_kind in {"progress", "stalled"}:
                key = (getattr(item, "job_id", ""), kind, getattr(payload, "index", 0))
                with self.lock:
                    self.progress[key] = item
                    if len(self.progress) > self.progress_limit:
                        self.progress.popitem(last=False)
                return
        self.control.put(item)

    def get_nowait(self):
        try:
            item = self.control.get_nowait()
            # Deliver pending progress before this video's terminal state.
            if isinstance(item, JobEvent) and item.kind == "video_progress":
                if getattr(item.payload, "kind", "") in {
                    "completed",
                    "failed",
                    "cancelled",
                    "skipped",
                }:
                    with self.lock:
                        self.progress.pop(
                            (item.job_id, item.kind, item.payload.index), None
                        )
            return item
        except Empty:
            with self.lock:
                if self.progress:
                    return self.progress.popitem(last=False)[1]
            raise

    def qsize(self):
        with self.lock:
            return self.control.qsize() + len(self.progress)


class JobController:
    def __init__(self, event_sink=None):
        self.state = JobState.IDLE
        self.context = None
        self.event_sink = event_sink
        self.completed = False
        self.close_when_finished = False

    @property
    def busy(self):
        return self.state not in {JobState.IDLE, JobState.FINISHED, JobState.CLOSED}

    def transition(self, state):
        if state not in _TRANSITIONS.get(self.state, set()):
            raise RuntimeError(
                f"不允許的工作狀態轉移：{self.state.value} → {state.value}"
            )
        self.state = state

    def begin(self, state):
        self.transition(state)
        self.context = TaskContext(event_sink=self.event_sink)
        self.completed = False
        return self.context

    def accepts(self, event):
        return (
            self.context is not None
            and event.job_id == self.context.job_id
            and not self.completed
        )

    def finish(self):
        if self.completed:
            return False
        self.transition(JobState.FINISHED)
        self.completed = True
        return True

    def cancel(self, *, close=False):
        if not self.busy or self.state == JobState.UPDATING:
            return
        self.close_when_finished = self.close_when_finished or close
        if self.state == JobState.CANCELLING:
            return
        self.transition(JobState.CANCELLING)
        self.context.cancellation.request()
        Thread(
            target=self.context.cancellation.cancel, name="job-cancel", daemon=True
        ).start()
