"""User-readable failure categories while retaining the original diagnostic."""

import errno
import time


def replace_with_retry(source, destination, *, cancellation=None):
    """Only Windows sharing/lock violations are transient enough to retry."""
    for attempt in range(3):
        if cancellation is not None and cancellation.cancelled:
            raise InterruptedError("檔案替換已取消。")
        try:
            return source.replace(destination)
        except OSError as error:
            if getattr(error, "winerror", None) not in {32, 33} or attempt == 2:
                raise
            if cancellation is not None:
                if cancellation.wait(0.05 * (attempt + 1)):
                    raise InterruptedError("檔案替換已取消。") from error
            else:
                time.sleep(0.05 * (attempt + 1))


def describe_error(error):
    if isinstance(error, InterruptedError):
        return "cancelled", "工作已取消。"
    if isinstance(error, OSError):
        code = error.errno
        if code == errno.ENOSPC:
            return "disk_full", f"磁碟空間不足，請釋放空間後重試。原始診斷：{error}"
        if getattr(error, "winerror", None) in {32, 33}:
            return "file_busy", f"檔案正由其他程式使用，請關閉後重試。原始診斷：{error}"
        if isinstance(error, PermissionError):
            return (
                "permission",
                f"沒有存取權限，請檢查來源與輸出資料夾。原始診斷：{error}",
            )
    if "來源影片" in str(error) and "變更" in str(error):
        return "source_changed", str(error)
    return type(error).__name__, str(error)
