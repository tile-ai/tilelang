"""Ascend-owned scheduling for graph compilation."""

from .schedule import UnsupportedScheduleError, schedule_primfunc

__all__ = ["UnsupportedScheduleError", "schedule_primfunc"]
