from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    if shape_x > 128 and shape_y > 128:
        raise ValueError("one-axis cutting is for a single large axis")
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
