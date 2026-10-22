from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    if max(shape_x, shape_y) > 128:
        raise ValueError("small-shape strategy is bounded by one 128x128 tile")
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
