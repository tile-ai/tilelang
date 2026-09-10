"""Loop-carried vector local.var updated inside T.serial, used after the loop.

This covers the low-level SimdVF pattern: allocate the accumulator
outside the serial loop, vmula each iteration, then store once. The test is
backend-agnostic and does not encode a product kernel.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


NUM_ITERS = 4
LANES = 64


def serial_vreg_accum_kernel(backend="asc"):
    @T.prim_func
    def main(
        A: T.Tensor((NUM_ITERS * LANES,), T.float32),
        B: T.Tensor((NUM_ITERS * LANES,), T.float32),
        Out: T.Tensor((LANES,), T.float32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((NUM_ITERS * LANES,), T.float32)
            b_ub = T.alloc_shared((NUM_ITERS * LANES,), T.float32)
            out_ub = T.alloc_shared((LANES,), T.float32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                acc = T.simd.alloc_var(T.float32)
                acc = T.simd.vdup(0.0, T.float32)
                for i in T.serial(NUM_ITERS):
                    a = T.simd.vld(a_ub[i * LANES])
                    b = T.simd.vld(b_ub[i * LANES])
                    T.simd.vmula(acc, a, b, mask)
                T.simd.vsts(out_ub[0], acc, mask)
            T.copy(out_ub, Out)

    return main


@pytest.mark.parametrize("backend", ["asc"])
def test_serial_vreg_accum(backend):
    kernel = tilelang.compile(serial_vreg_accum_kernel(backend), target=backend, out_idx=[2])
    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(NUM_ITERS * LANES, dtype=torch.float32, device=device)
    b = torch.randn(NUM_ITERS * LANES, dtype=torch.float32, device=device)
    out = kernel(a, b)
    torch.npu.synchronize()

    expected = (a.view(NUM_ITERS, LANES) * b.view(NUM_ITERS, LANES)).sum(dim=0)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
