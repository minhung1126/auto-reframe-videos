# -*- coding: utf-8 -*-
"""Unified command-line entry for GUI, Reframe, and Compress modes."""

import argparse
from collections.abc import Sequence

from auto_reframe_core.version import __version__


MODES = ("gui", "reframe", "compress")


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "gui":
        from auto_reframe_core.gui import main as run
    elif args.mode == "reframe":
        from auto_reframe_core.reframe import main as run
    else:
        from auto_reframe_core.compress import main as run

    result = run()
    return result if isinstance(result, int) else 0
