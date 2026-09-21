from .broadcast_add_kernel import build as _build


def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    """Build row-wise x[row, col] + bias[row] with row ownership per core."""
    return _build(rows, cols, dtype, num_cores)
