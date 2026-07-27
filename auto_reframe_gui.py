# -*- coding: utf-8 -*-
"""Backward-compatible GUI entry point.

Use ``python -m auto_reframe_core`` for the unified application entry.
"""

import sys

from auto_reframe_core import gui as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
