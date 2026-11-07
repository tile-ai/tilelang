from .softmax_v2_tiling_data import SoftmaxTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> SoftmaxTiling:
    if cols <= 4096:
        strategy, tile_cols = "full_load", cols
    elif cols <= 16384:
        strategy, tile_cols = "recompute", 4096
    else:
        strategy, tile_cols = "online", 4096
    return SoftmaxTiling(rows, cols, tile_cols, num_cores, strategy)
