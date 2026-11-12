from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    if shape_x <= 128 or shape_y <= 128:
        raise ValueError("two-axis cutting requires both axes larger than one tile")
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
