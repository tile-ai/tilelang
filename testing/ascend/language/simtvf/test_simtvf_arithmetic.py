"""Packed FP8 carriers must use per-lane floating-point arithmetic."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T

THREADS = 128


@tilelang.jit(out_idx=[2], target="ascend")
def fp8_vecadd_kernel(n: int, dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            b_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=THREADS):
                a_local = T.alloc_fragment((n,), dtype)
                b_local = T.alloc_fragment((n,), dtype)
                c_local = T.alloc_fragment((n,), dtype)

                T.copy(a_ub, a_local)
                T.copy(b_ub, b_local)
                for i in T.Parallel(n):
                    c_local[i] = a_local[i] + b_local[i]
                T.copy(c_local, c_ub)
            T.copy(c_ub, C)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16], ids=lambda n: f"lanes-{n}")
def test_packed_fp8_add(dtype, lanes):
    n = THREADS * lanes
    td = getattr(torch, dtype)
    a = torch.linspace(-128, 128, n).to(td)
    b = torch.linspace(64, -64, n).to(td)
    actual = fp8_vecadd_kernel(n, dtype)(a.npu(), b.npu()).cpu().float()
    expected = (a.float() + b.float()).to(td).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
