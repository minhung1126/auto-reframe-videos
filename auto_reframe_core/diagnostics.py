"""Rotating local diagnostics and deliberately data-minimal export reports."""

import json
import logging
from logging.handlers import RotatingFileHandler
import platform
import sys

from auto_reframe_core.version import __version__


def local_logger():
    from auto_reframe_core.runtime_paths import logs_root

    logger = logging.getLogger("auto_reframe.jobs")
    if not logger.handlers:
        handler = RotatingFileHandler(
            logs_root() / "jobs.log",
            maxBytes=1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def diagnostic_report(result=None):
    # Never export source paths, filenames, overlays, raw stderr or config contents.
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    if result is not None:
        report["result"] = {
            "status": result.status.value,
            "videos": len(result.videos),
            "counts": {
                status: result.count(status)
                for status in ("success", "skipped", "failed", "partial", "cancelled")
            },
            "error_kinds": sorted(
                {video.error_kind for video in result.videos if video.error_kind}
            ),
        }
    return report


def export_diagnostics(path, result=None):
    from pathlib import Path

    Path(path).write_text(
        json.dumps(diagnostic_report(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
