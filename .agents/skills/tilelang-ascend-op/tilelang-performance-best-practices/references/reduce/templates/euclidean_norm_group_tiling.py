from .euclidean_norm_tiling import select_tiling


def select_group_tiling(groups: int, rows_per_group: int, cols: int, num_cores: int = 64):
    if groups <= 0 or rows_per_group <= 0:
        raise ValueError("group dimensions must be positive")
    return select_tiling(groups * rows_per_group, cols, num_cores)
