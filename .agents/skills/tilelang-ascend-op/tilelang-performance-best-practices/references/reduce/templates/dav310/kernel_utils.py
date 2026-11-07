import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=-1)
def euclidean_norm(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    # Match the production Ascend Norm path and TileLang RMSNorm example:
    # infer the SIMT fragment on a power-of-two extent and guard valid lanes.
    padded_cols = tilelang.next_power_of_2(cols)
    if padded_cols > 4096:
        raise NotImplementedError("the executable full-load baseline supports padded cols <= 4096; use a verified split-R kernel")
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer from target hardware discovery")
    active = min(rows, num_cores)

    @T.prim_func
    def kernel(x: T.Buffer((rows, cols), dtype), out: T.Buffer((rows,), "float32")):
        with T.Kernel(active) as core_id:
            x_ub = T.alloc_shared((padded_cols,), dtype)
            for row_iter in T.serial((rows + active - 1) // active):
                row = row_iter * active + core_id
                if row < rows:
                    T.copy(x[row, :], x_ub[:cols])
                    with T.SimtVF(threads=128):
                        squares = T.alloc_fragment((padded_cols,), "float32")
                        total = T.alloc_fragment((1,), "float32")
                        for i in T.Parallel(padded_cols):
                            if i < cols:
                                squares[i] = T.cast(x_ub[i], "float32") * T.cast(x_ub[i], "float32")
                            else:
                                squares[i] = T.float32(0)
                        T.reduce_sum(squares, total, dim=0)
                        out[row] = T.sqrt(total[0])

    return kernel


@tilelang.jit(out_idx=-1)
def softmax_full_load(rows: int, cols: int, dtype: str = "float16", num_cores: int | None = None):
    """Numerically stable full-load reference; exp and sum stay fp32."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if cols > 4096:
        raise NotImplementedError("the executable full-load baseline supports cols <= 4096; use a verified recompute/online kernel")
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer from target hardware discovery")
    active = min(rows, num_cores)

    @T.prim_func
    def kernel(x: T.Buffer((rows, cols), dtype), out: T.Buffer((rows, cols), dtype)):
        with T.Kernel(active) as core_id:
            x_ub = T.alloc_shared((cols,), dtype)
            exp_ub = T.alloc_shared((cols,), "float32")
            out_ub = T.alloc_shared((cols,), dtype)
            for row_iter in T.serial((rows + active - 1) // active):
                row = row_iter * active + core_id
                if row < rows:
                    T.copy(x[row, :], x_ub)
                    with T.SimtVF(threads=1):
                        maximum = T.alloc_fragment((1,), "float32")
                        total = T.alloc_fragment((1,), "float32")
                        maximum[0] = T.cast(x_ub[0], "float32")
                        for i in T.serial(cols):
                            value = T.cast(x_ub[i], "float32")
                            if value > maximum[0]:
                                maximum[0] = value
                        total[0] = T.float32(0)
                        for i in T.serial(cols):
                            exp_ub[i] = T.exp(T.cast(x_ub[i], "float32") - maximum[0])
                            total[0] = total[0] + exp_ub[i]
                        for i in T.serial(cols):
                            out_ub[i] = T.cast(exp_ub[i] / total[0], dtype)
                    T.copy(out_ub, out[row, :])

    return kernel


# Historical callers use this name for the executable full-load baseline.
softmax = softmax_full_load
