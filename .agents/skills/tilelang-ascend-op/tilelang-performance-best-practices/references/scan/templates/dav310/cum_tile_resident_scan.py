from .scan_base import build as _baseline
from ..scan_tiling_data import ScanStrategy

STATUS = "EXECUTABLE_BASELINE"
STRATEGY = ScanStrategy("tile_resident", "one core owns one complete row", True, "O(R) baseline", "the complete row and fp32 output fit UB")


def build(rows: int, cols: int, num_cores: int | None = None):
    return _baseline(rows, cols, cols, num_cores)
