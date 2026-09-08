"""Communication-aware column ownership and intermediate vector widths."""

import tilelang.language as T


def make_reducer(rows=8, columns=128, width=None, dtype="float32", op="sum", updates=1, repeats=1, batch=1, total=False, seed=None):
    width = None if width is None else T.int32(width)
    outputs = 1 if total else columns

    @T.prim_func
    def main(inputs: T.Tensor((rows, columns), dtype), output: T.Tensor((outputs,), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((rows, columns), dtype)
            values = T.alloc_fragment((rows, columns), dtype)
            acc = T.alloc_reducer((outputs,), dtype, op=op)
            result = T.alloc_fragment((outputs,), dtype)
            T.copy(inputs, shared)
            T.copy(shared, values)
            T.reducer_init(acc, init=seed)
            for _repeat in T.serial(repeats):
                for row, column in T.Parallel(rows, columns, coalesced_width=width):
                    T.reducer_update(acc[0 if total else column], values[row, column])
                if updates > 1:
                    for row, column in T.Parallel(rows, columns, coalesced_width=width):
                        T.reducer_update(acc[0 if total else column], values[row, column])
            T.finalize_reducer(acc, result, batch=batch)
            T.copy(result, output)

    return main


def make_communication_reducer(within_warp=True):
    @T.prim_func
    def main(inputs: T.Tensor((8, 128), "float32"), output: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((8, 128), "float32")
            acc = T.alloc_reducer((128,), "float32")
            result = T.alloc_fragment((128,), "float32")
            T.annotate_layout(
                {
                    values: T.Fragment(
                        (8, 128),
                        forward_thread_fn=lambda row, column: (column // 4 * 4 + row % 4) if within_warp else (row % 4 * 32 + column // 4),
                        forward_index_fn=lambda row, column: row // 4 * 4 + column % 4,
                    )
                }
            )
            T.copy(inputs, values)
            T.reducer_init(acc)
            for row, column in T.Parallel(8, 128):
                T.reducer_update(acc[column], values[row, column])
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    return main


VARIANTS = {
    "columns": lambda: make_reducer(),
    "intermediate": lambda: make_reducer(rows=4, columns=256),
    "full": lambda: make_reducer(rows=4, columns=256, total=True),
    "warp_collective": make_communication_reducer,
}


def check(variant, model, result):
    if variant == "full":
        assert result["buffers"]["acc"]["replicate"] == 128
    elif variant == "warp_collective":
        assert result["buffers"]["acc"]["replicate"] == 4
    elif model == "register-count":
        assert result["buffers"]["acc"]["replicate"] == 1
