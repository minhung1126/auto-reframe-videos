"""Kernel-held file locks, released by the OS even after a process crashes."""

from contextlib import ExitStack
from hashlib import sha256
import os
from pathlib import Path
import time
import unicodedata

from auto_reframe_core.platform_profile import current_platform


class OutputBusyError(RuntimeError):
    pass


def path_identity(path):
    value = str(Path(path).expanduser().resolve())
    if current_platform().case_fold_output_paths:
        return unicodedata.normalize("NFC", value).casefold()
    return os.path.normcase(value)


class FileLock:
    def __init__(self, path, *, label=None, timeout=0):
        self.path = Path(path)
        self.label = label or str(path)
        self.timeout = timeout
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never unlink a lock file: doing so creates a second, unlocked inode.
        self.stream = self.path.open("a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if current_platform().is_windows:
                    import msvcrt

                    if self.path.stat().st_size == 0:
                        self.stream.write(b"\0")
                        self.stream.flush()
                    self.stream.seek(0)
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if time.monotonic() >= deadline:
                    self.stream.close()
                    self.stream = None
                    raise OutputBusyError(f"其他工作正在使用：{self.label}") from exc
                time.sleep(0.025)

    def __exit__(self, *_):
        if self.stream is not None:
            try:
                if current_platform().is_windows:
                    import msvcrt

                    self.stream.seek(0)
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


class OutputLocks:
    def __init__(self, finals):
        paths = {path_identity(path): Path(path) for path in finals}
        self.paths = tuple(paths[key] for key in sorted(paths))
        self.stack = ExitStack()

    def acquire(self):
        try:
            for path in self.paths:
                digest = sha256(path_identity(path).encode("utf-8")).hexdigest()
                self.stack.enter_context(
                    FileLock(path.parent / f".arv-output-{digest}.lock", label=path)
                )
        except BaseException:
            self.release()
            raise
        return self

    def release(self):
        self.stack.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
