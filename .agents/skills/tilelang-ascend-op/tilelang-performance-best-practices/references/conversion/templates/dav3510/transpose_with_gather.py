from .transpose_base import build as _build


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    """Select the SIMD vgather2 transpose used for 16/32-bit element types."""
    if dtype.bytes not in (2, 4):
        raise ValueError("indexed gather path supports 16/32-bit elements")
    return _build(shape_x, shape_y, dtype, kernel_factory=kernel_factory)
