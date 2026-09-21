import tilelang
from tilelang import language as T

from .broadcast_add_tiling import select_tiling

STATUS = "EXECUTABLE_BASELINE"


@tilelang.jit(out_idx=-1)
def build(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer from target hardware discovery")
    cfg = select_tiling(rows, cols, T.dtype(dtype).bytes, num_cores)
    tiles = (cols + cfg.tile_cols - 1) // cfg.tile_cols

    @T.prim_func
    def kernel(a: T.Buffer((rows, cols), dtype), b: T.Buffer((rows, 1), dtype), out: T.Buffer((rows, cols), dtype)):
        with T.Kernel(cfg.active_cores) as core_id:
            a_ub = T.alloc_shared((cfg.tile_cols,), dtype)
            out_ub = T.alloc_shared((cfg.tile_cols,), dtype)
            scalar = T.alloc_shared((1,), dtype)
            for row_iter in T.serial((rows + cfg.active_cores - 1) // cfg.active_cores):
                row = row_iter * cfg.active_cores + core_id
                if row < rows:
                    T.copy(b[row, 0], scalar[0])
                    for tile in T.serial(tiles):
                        begin = tile * cfg.tile_cols
                        valid = T.min(cfg.tile_cols, cols - begin)
                        T.copy(a[row, begin:begin + valid], a_ub[:valid])
                        with T.SimtVF(threads=128):
                            a_fp32 = T.alloc_fragment((cfg.tile_cols,), "float32")
                            bias_fp32 = T.alloc_fragment((1,), "float32")
                            bias_fp32[0] = T.cast(scalar[0], "float32")
                            for i in T.Parallel(cfg.tile_cols):
                                if i < valid:
                                    a_fp32[i] = T.cast(a_ub[i], "float32")
                                    out_ub[i] = T.cast(a_fp32[i] + bias_fp32[0], dtype)
                        T.copy(out_ub[:valid], out[row, begin:begin + valid])
    return kernel
