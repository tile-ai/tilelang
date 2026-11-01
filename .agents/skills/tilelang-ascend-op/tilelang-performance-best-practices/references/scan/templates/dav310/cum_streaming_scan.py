from ..scan_tiling_data import ScanStrategy

STATUS = "EXECUTABLE_BASELINE"
STRATEGY = ScanStrategy("streaming", "one core owns one complete row", False, "O(R)", "fp32 carry remains resident across tiles")
