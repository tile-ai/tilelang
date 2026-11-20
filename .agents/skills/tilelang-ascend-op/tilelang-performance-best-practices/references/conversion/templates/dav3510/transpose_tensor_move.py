from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    """Use the verified GM->UB->transpose->GM PTO data-movement path."""
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
