# -*- coding: utf-8 -*-
"""Backward-compatible Reframe entry point.

Use ``python -m auto_reframe_core reframe`` for the unified application entry.
"""

import sys

from auto_reframe_core import reframe as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
