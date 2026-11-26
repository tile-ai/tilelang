import tilelang
from tilelang import language as T

STATUS = "EXECUTABLE_BASELINE"


@tilelang.jit(out_idx=-1)
def build(rows: int, cols: int, tile_cols: int = 256, num_cores: int | None = None):
    """Correctness-first PTO inclusive scan over the contiguous last axis.

    One kernel task owns each row, so the carry never crosses cores. The
    arithmetic state remains float32 and the result is cast only at the store.
    Tune tile_cols and replace the serial inner scan only after a PTO SIMD
    lane-scan microbenchmark passes for every tail class.
    """

    if min(rows, cols, tile_cols) <= 0:
        raise ValueError("rows, cols, and tile_cols must be positive")
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer from target hardware discovery")
    active_cores = min(num_cores, rows)
    tiles = (cols + tile_cols - 1) // tile_cols

    @T.prim_func
    def kernel(
        x: T.Buffer((rows, cols), "bfloat16"),
        out: T.Buffer((rows, cols), "float32"),
    ):
        with T.Kernel(active_cores) as core_id:
            x_ub = T.alloc_shared((tile_cols,), "bfloat16")
            out_ub = T.alloc_shared((tile_cols,), "float32")
            carry_ub = T.alloc_shared((1,), "float32")

            for row_iter in T.serial((rows + active_cores - 1) // active_cores):
                row = row_iter * active_cores + core_id
                if row < rows:
                    carry_ub[0] = T.float32(0)
                    for tile in T.serial(tiles):
                        begin = tile * tile_cols
                        valid = T.min(tile_cols, cols - begin)
                        T.copy(x[row, begin : begin + valid], x_ub[:valid])

                        with T.SimtVF(threads=1):
                            running = T.alloc_fragment((1,), "float32")
                            running[0] = carry_ub[0]
                            for i in T.serial(tile_cols):
                                if i < valid:
                                    running[0] = running[0] + T.cast(x_ub[i], "float32")
                                    out_ub[i] = running[0]
                            carry_ub[0] = running[0]

                        T.copy(out_ub[:valid], out[row, begin : begin + valid])

    return kernel


def reference(x):
    return x.float().cumsum(dim=-1)
