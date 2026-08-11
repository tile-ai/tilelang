import torch

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

BC, BK, NB = 16, 64, 16
MESSAGE = "Uninitialized T.gemm accumulator"


def _make(init_mode):
    """Build the #2936 kernel. init_mode: 'none' | 'clear' | 'clear_accum'."""

    @T.prim_func
    def kernel(
        A: T.Tensor((NB, BC, BK), torch.float32),
        B: T.Tensor((NB, BK, BC), torch.float32),
        C: T.Tensor((NB, BC, BC), torch.float32),
    ):
        with T.Kernel(1, NB, threads=64) as (bx, by):
            bn = bx * NB + by
            a_s = T.alloc_shared((BC, BK), torch.float32)
            b_s = T.alloc_shared((BK, BC), torch.float32)
            T.copy(A[bn, :, :], a_s)
            T.copy(B[bn, :, :], b_s)
            c_f = T.alloc_fragment((BC, BC), torch.float32)
            if init_mode == "clear":
                T.clear(c_f)
            T.gemm(a_s, b_s, c_f, clear_accum=(init_mode == "clear_accum"))
            T.copy(c_f, C[bn, :, :])

    return tvm.IRModule.from_expr(kernel)


def _verify(mod, capfd):
    tl.transform.VerifyGemmAccumInit()(mod)
    return capfd.readouterr().err


def test_warns_when_accumulator_is_uninitialized(capfd):
    assert MESSAGE in _verify(_make("none"), capfd)


def test_silent_when_cleared_with_t_clear(capfd):
    assert MESSAGE not in _verify(_make("clear"), capfd)


def test_silent_when_clear_accum_is_true(capfd):
    assert MESSAGE not in _verify(_make("clear_accum"), capfd)


if __name__ == "__main__":
    tilelang.testing.main()
