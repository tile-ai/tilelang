from .euclidean_norm_tiling_data import EuclideanNormTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> EuclideanNormTiling:
    if rows <= 0 or cols <= 0 or num_cores <= 0:
        raise ValueError("rows, cols, and num_cores must be positive")
    padded_cols = 1 << (cols - 1).bit_length()
    tile_cols = min(padded_cols, 4096)
    strategy = "padded_full_load" if padded_cols <= 4096 else "split_r_design_only"
    return EuclideanNormTiling(rows, cols, tile_cols, num_cores, strategy)
