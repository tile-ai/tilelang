from .kernel_utils import softmax

STATUS = "EXECUTABLE_BASELINE"


def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    return softmax(rows, cols, dtype, num_cores)
