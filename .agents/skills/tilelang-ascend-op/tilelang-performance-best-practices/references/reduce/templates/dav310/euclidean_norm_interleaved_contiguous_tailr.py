from .kernel_utils import euclidean_norm

STATUS = "EXECUTABLE_BASELINE"


def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    return euclidean_norm(rows, cols, dtype, num_cores)
