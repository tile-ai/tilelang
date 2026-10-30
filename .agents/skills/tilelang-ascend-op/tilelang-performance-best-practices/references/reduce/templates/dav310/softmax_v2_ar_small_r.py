from .kernel_utils import softmax_full_load
from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "PARTIAL"
STRATEGY = SoftmaxStrategy("ar_small_r", "[A, R]", 1, ("max", "sum", "exp"), "small R; batch multiple rows per task after benchmarking")


def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    return softmax_full_load(rows, cols, dtype, num_cores)
