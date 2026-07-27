"""Backward-compatible import alias for shared video utilities."""

import sys

from auto_reframe_core import video_utils as _implementation


sys.modules[__name__] = _implementation
