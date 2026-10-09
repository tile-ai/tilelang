from .broadcast_common import BroadcastTiling


def select_tiling(rows: int, cols: int, dtype_bytes: int, num_cores: int = 64) -> BroadcastTiling:
    ub_budget = 96 * 1024
    vectors = max(1, ub_budget // (3 * dtype_bytes * 256))
    # PTO SIMT layout inference requires a complete vector/thread footprint.
    # Keep the GM extent in ``valid`` while padding the UB/fragment extent.
    padded_cols = ((cols + 255) // 256) * 256
    tile_cols = min(padded_cols, vectors * 256)
    cfg = BroadcastTiling(rows, cols, tile_cols, num_cores)
    cfg.validate()
    return cfg
