"""Adapt an explicitly supplied, validated transpose factory from the current repository.

The factory receives full shape_x, shape_y and dtype and returns the callable
kernel. Device validation and the factory's input/output contract remain the
caller's responsibility; this adapter does not implement a transpose kernel.
"""

from ..transpose_tiling_data import TransposeTiling, select_tiling

STATUS = "PARTIAL"


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    cfg = select_tiling(shape_x, shape_y)
    cfg.validate(shape_x, shape_y)
    if not callable(kernel_factory):
        raise ValueError("Pass a validated kernel_factory(shape_x, shape_y, dtype) from the current repository")
    return kernel_factory(shape_x, shape_y, dtype)


__all__ = ["TransposeTiling", "build"]
