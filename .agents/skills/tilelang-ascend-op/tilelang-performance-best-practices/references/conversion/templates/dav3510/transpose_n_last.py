from .transpose_base import build as _build


def build(batch: int, m: int, n: int, dtype, *, kernel_factory=None):
    if batch < 1:
        raise ValueError("batch must be positive")
    return _build(m, n, dtype, kernel_factory=kernel_factory)
