from dataclasses import replace

from ..scan_tiling import select_tiling


def select_cummin_tiling(rows: int, cols: int, num_cores: int = 64):
    return replace(select_tiling(rows, cols, num_cores), strategy="cummin_streaming")
