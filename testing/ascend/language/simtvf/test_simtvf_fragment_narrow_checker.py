import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_to_host_device_ir


def _run_to_optimize(func):
    context = create_backend_context("ascend")
    mod = tvm.IRModule({"main": func})
    return lower_to_host_device_ir(mod, context)


def test_fragment_narrow_checker_fragment():
    @T.prim_func
    def narrowed(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=128):
            frag = T.alloc_fragment((16,), "float32")
            for i in T.Parallel(16):
                frag[i] = A[i]
                B[i] = frag[i] + T.float32(1)

    _run_to_optimize(narrowed)


def test_fragment_narrow_checker_rejects_fragment_leak():
    @T.prim_func
    def leaked(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1):
            frag = T.alloc_fragment((16,), "float32")
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    frag[i] = A[i]
            B[0] = frag[0]

    with pytest.raises(ValueError, match="Parallel loops outside VF blocks are not supported on Ascend NPU"):
        _run_to_optimize(leaked)


test_fragment_narrow_checker_rejects_fragment_leak()
