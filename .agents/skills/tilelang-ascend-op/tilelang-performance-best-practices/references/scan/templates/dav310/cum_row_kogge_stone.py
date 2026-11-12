from ..scan_tiling_data import ScanStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = ScanStrategy(
    "row_kogge_stone", "one core owns one resident row", True, "ceil(log2(R))", "static R; every level combines element i with i-2^level"
)


def level_pairs(length: int, level: int):
    distance = 1 << level
    return [(i - distance, i) for i in range(distance, length)]
