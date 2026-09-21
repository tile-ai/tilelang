from ..scan_tiling_data import ScanStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = ScanStrategy("lane_parallel", "one core owns one row tile", True, "O(log lanes)", "PTO SIMD shift/gather microtest must pass for every tail class")
