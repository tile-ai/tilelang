"""Compatibility exports for the pass-timing developer tool.

The implementation lives in :mod:`tilelang.tools.pass_timing`.  This module
preserves the historical import path for downstream users.
"""

from tilelang.tools.pass_timing import (
    PassTimingRecord,
    PassTimingTool,
    TileLangPassTimingInstrument,
    create_pass_timing_tool,
)

__all__ = [
    "PassTimingRecord",
    "PassTimingTool",
    "TileLangPassTimingInstrument",
    "create_pass_timing_tool",
]
