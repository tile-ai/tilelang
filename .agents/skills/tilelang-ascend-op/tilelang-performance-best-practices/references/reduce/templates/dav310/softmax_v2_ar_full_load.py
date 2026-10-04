from .kernel_utils import softmax_full_load
from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "EXECUTABLE_BASELINE"
STRATEGY = SoftmaxStrategy("ar_full_load", "[A, R]", 1, ("max", "sum", "exp"), "one row and fp32 workspace fit UB")


def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    return softmax_full_load(rows, cols, dtype, num_cores)
