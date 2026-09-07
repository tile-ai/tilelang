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


VARIANTS = {
    "columns": lambda: make_reducer(),
    "intermediate": lambda: make_reducer(rows=4, columns=256),
    "full": lambda: make_reducer(rows=4, columns=256, total=True),
}


def check(variant, model, result):
    if variant == "full":
        assert result["buffers"]["acc"]["replicate"] == 128
    elif model == "register-count":
        assert result["buffers"]["acc"]["replicate"] == 1
