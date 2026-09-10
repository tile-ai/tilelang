"""Scalar column ownership versus a necessary full-participant reduction."""

import tilelang.language as T


def make_reducer(total=False, width=None, pinned=False, pin_loop=False, updates=1):
    width = None if width is None else T.int32(width)
    rows, cols = (4, 256) if total else (8, 128)
    outputs = 1 if total else cols
    dtype = T.bfloat16 if total else T.float32
    if total:
        layout = T.Fragment(
            (rows, cols),
            forward_thread_fn=lambda i, j: (i * cols + j) // 8,
            forward_index_fn=lambda i, j: j % 8,
        )
    else:
        layout = T.Fragment(
            (rows, cols),
            forward_thread_fn=lambda i, j: (i % 4) * 32 + j // 4,
            forward_index_fn=lambda i, j: (i // 4) * 4 + j % 4,
        )

    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((outputs,), T.float32)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((rows, cols), dtype)
            values = T.alloc_fragment((rows, cols), T.float32)
            if pinned:
                T.annotate_layout({values: layout})
            T.copy(A, shared)
            T.copy(shared, values)
            acc = T.alloc_reducer((outputs,), T.float32)
            result = T.alloc_fragment((outputs,), T.float32)
            T.reducer_init(acc)
            if updates > 0:
                for i, j in T.Parallel(rows, cols, coalesced_width=width, loop_layout=layout if pin_loop else None):
                    T.reducer_update(acc[0 if total else j], values[i, j])
            if updates > 1:
                for i, j in T.Parallel(rows, cols):
                    T.reducer_update(acc[0 if total else j], values[i, j])
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return main


VARIANTS = {
    "columns": lambda: make_reducer(),
    "full_bf16": lambda: make_reducer(total=True),
}


def check(variant, model, result):
    if variant == "columns" and model == "register-count":
        assert result["buffers"]["acc"]["replicate"] == 1
    if variant == "full_bf16":
        assert result["buffers"]["acc"]["replicate"] == 128


# This case's answers depend on the *build configuration*, not only on the
# pinned target: the `sm_90` goldens were recorded against a build with CUDA
# codegen enabled, and a build without it disagrees. Concretely, a
# `-DUSE_CUDA=OFF` build registers a different `tl.copy` implementation for
# `sm_90` (src/cuda/op/copy.cc is only compiled into the CUDA backend), and
# every variant here then comes out fully replicated -- including the
# `width=4` variant of #3171's unit test, which returns `combine_size == 128`
# instead of the 4 it asserts. On the CUDA-enabled build the same assertions
# pass, which is where these goldens are meaningful.
#
# So run this suite against a CUDA-enabled build (the CI gate does). A
# non-CUDA build reporting drift here is a build-capability mismatch, not a
# layout regression.
