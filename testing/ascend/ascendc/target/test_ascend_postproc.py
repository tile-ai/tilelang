"""The Ascend source hook applies during lowering and is isolated per test."""

import tilelang
import tilelang.ascend.language as T
import tvm_ffi
from tilelang.ascend.callback import register_ascend_postproc_callback


def test_postproc_callback():
    @T.prim_func
    def kernel(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(64):
                B[i] = A[i] + 1

    name = "tilelang_callback_ascend_postproc"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    seen_targets = []
    marker = "// ASCEND_POSTPROC_TEST"
    try:

        @register_ascend_postproc_callback
        def postproc(code, target):
            seen_targets.append(target.kind.name)
            return marker + "\n" + code

        with tilelang.tvm.target.Target("ascend"):
            source = tilelang.lower(kernel, target="ascend").kernel_source
        assert marker in source
        assert seen_targets == ["ascend"]
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)
