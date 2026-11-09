"""Shared child-process ownership, cancellation, timeouts and pipe draining."""

from collections import deque
from pathlib import Path
from queue import Queue, Empty, Full
import subprocess
import threading
import time

from auto_reframe_core.platform_profile import hidden_subprocess_kwargs


class TaskCancelled(InterruptedError):
    """A user stopped a managed job, rather than a media-processing failure."""


def terminate_process(process, timeout=3.0):
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()


class ProcessCancellation:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes = set()

    @property
    def cancelled(self):
        return self._event.is_set()

    @property
    def active_process_count(self):
        with self._lock:
            return len(self._processes)

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def check(self):
        if self.cancelled:
            raise TaskCancelled("工作已取消。")

    def request(self):
        """Immediate, nonblocking feedback; process reaping runs off the UI thread."""
        self._event.set()

    def register(self, process):
        with self._lock:
            self._processes.add(process)
            cancelled = self.cancelled
        if cancelled:
            terminate_process(process)
            self.unregister(process)

    def unregister(self, process):
        with self._lock:
            self._processes.discard(process)

    def cancel(self, timeout=3.0):
        self.request()
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        deadline = time.monotonic() + max(0.0, timeout)
        for process in processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
        for process in processes:
            process.wait()
            self.unregister(process)


class ProcessRunner:
    def __init__(self, context=None):
        self.context = context
        self.cancellation = context.cancellation if context is not None else None

    def check(self):
        if self.cancellation is not None:
            self.cancellation.check()

    def capture(self, command, *, timeout=30, **kwargs):
        """Compatibility calls still receive subprocess.run's CompletedProcess."""
        self.check()
        if self.cancellation is None:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **kwargs,
            )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        self.cancellation.register(process)
        deadline = time.monotonic() + timeout
        try:
            while True:
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    self.check()
                    return subprocess.CompletedProcess(
                        command, process.returncode, stdout, stderr
                    )
                except subprocess.TimeoutExpired:
                    continue
        finally:
            terminate_process(process, timeout=0.5)
            self.cancellation.unregister(process)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()

    def ffmpeg(
        self,
        command,
        info,
        progress=None,
        *,
        status=None,
        debug_path=None,
        stall_after=30,
    ):
        """Read -progress and stderr independently; inactivity is observable, never fatal."""
        self.check()
        command = list(command)
        if Path(command[0]).stem.lower() == "ffmpeg":
            command[1:1] = ["-nostdin", "-nostats", "-progress", "pipe:1"]
        debug = Path(debug_path).open("w", encoding="utf-8") if debug_path else None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_subprocess_kwargs(),
            )
        except BaseException:
            if debug is not None:
                debug.close()
            raise
        if self.cancellation is not None:
            self.cancellation.register(process)
        lines = Queue(maxsize=256)
        stderr = deque(maxlen=30)
        heartbeat = [time.monotonic()]
        reader_done = threading.Event()
        stop_reading = threading.Event()
        reader_errors = []

        def read_progress():
            try:
                for line in process.stdout:
                    heartbeat[0] = time.monotonic()
                    while not stop_reading.is_set():
                        try:
                            lines.put(line, timeout=0.05)
                            break
                        except Full:
                            continue
            finally:
                reader_done.set()

        def read_stderr():
            try:
                for line in process.stderr:
                    heartbeat[0] = time.monotonic()
                    stderr.append(line)
                    if debug is not None:
                        debug.write(line)
                        debug.flush()
            except OSError as error:
                reader_errors.append(error)

        readers = [
            threading.Thread(target=read_progress, daemon=True),
            threading.Thread(target=read_stderr, daemon=True),
        ]
        for reader in readers:
            reader.start()
        last_percent = -1
        stalled = False
        try:
            if progress:
                progress(0)
                last_percent = 0
            while (
                not reader_done.is_set() or not lines.empty() or process.poll() is None
            ):
                if reader_errors:
                    raise reader_errors[0]
                if self.cancellation is not None and self.cancellation.cancelled:
                    terminate_process(process, timeout=0.5)
                try:
                    line = lines.get(timeout=0.05).strip()
                except Empty:
                    if not stalled and time.monotonic() - heartbeat[0] >= stall_after:
                        stalled = True
                        if status:
                            status("暫未收到進度；仍在等待，可隨時停止")
                    continue
                if stalled:
                    stalled = False
                    if status:
                        status("已恢復處理進度")
                key, _, value = line.partition("=")
                if key in {"out_time_us", "out_time_ms"}:
                    try:
                        duration = float(info.get("duration", 0))
                        percent = (
                            min(100, max(0, int(float(value) / 10000 / duration)))
                            if duration
                            else 0
                        )
                    except (ValueError, TypeError):
                        continue
                    if percent > last_percent and progress:
                        last_percent = percent
                        progress(percent)
            process.wait()
            if process.returncode == 0 and progress and last_percent < 100:
                progress(100)
            return process.returncode, list(stderr)
        finally:
            stop_reading.set()
            terminate_process(process, timeout=0.5)
            for reader in readers:
                reader.join(timeout=1)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            if debug is not None:
                debug.close()
            if self.cancellation is not None:
                self.cancellation.unregister(process)
