# -*- coding: utf-8 -*-
"""Unified command-line entry for GUI, Reframe, and Compress modes."""

import argparse
import sys
from collections.abc import Sequence

from auto_reframe_core.version import __version__


MODES = ("gui", "reframe", "compress")


def configure_utf8_stdio() -> None:
    """Keep localized CLI output writable on Windows and redirected streams."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_reframe_core",
        description="Auto Reframe Videos 統一入口",
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        default="gui",
        nargs="?",
        help="執行模式；省略時啟動 GUI（預設：gui）",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--desktop-smoke", metavar="REPORT", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from auto_reframe_core.runtime_paths import is_frozen, logs_root
    if is_frozen():
        # Windowed PyInstaller builds have no standard streams.
        for name in ("stdout", "stderr"):
            if getattr(sys, name) is None:
                setattr(sys, name, (logs_root() / "application.log").open("a", encoding="utf-8", buffering=1))
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    if args.desktop_smoke:
        from auto_reframe_core.desktop_smoke import run_desktop_smoke
        return run_desktop_smoke(args.desktop_smoke)

    if is_frozen():
        from auto_reframe_core.platform_profile import register_installer_mutex
        register_installer_mutex()
    if args.mode == "gui":
        from auto_reframe_core.gui import main as run
    elif args.mode == "reframe":
        from auto_reframe_core.reframe import main as run
    else:
        from auto_reframe_core.compress import main as run

    result = run()
    return result if isinstance(result, int) else 0
