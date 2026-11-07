from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    if max(shape_x, shape_y) < 4096:
        raise ValueError("big-dimension strategy starts at 4096 elements")
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
