from .scan_tiling_data import ScanTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> ScanTiling:
    if cols <= 256:
        strategy, tile_cols = "tile_resident", cols
    elif rows >= num_cores:
        strategy, tile_cols = "streaming", 256
    else:
        strategy, tile_cols = "core_partition", 256
    return ScanTiling(rows, cols, tile_cols, num_cores, strategy)
