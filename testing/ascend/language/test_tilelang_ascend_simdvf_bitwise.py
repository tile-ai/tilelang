"""Tests for T.simd.vxor and T.simd.vnot on Ascend SimdVF."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def bitwise_kernel(n, backend="asc"):
    @T.prim_func
    def main(
        A: T.Buffer((n,), "int32"),
        B: T.Buffer((n,), "int32"),
        Xor: T.Buffer((n,), "int32"),
        Not: T.Buffer((n,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            b_ub = T.alloc_shared((n,), "int32")
            xor_ub = T.alloc_shared((n,), "int32")
            not_ub = T.alloc_shared((n,), "int32")

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                T.simd.vsts(xor_ub[0], T.simd.vxor(a, b), mask)
                T.simd.vsts(not_ub[0], T.simd.vnot(a), mask)
            T.copy(xor_ub, Xor)
            T.copy(not_ub, Not)

    return main


@pytest.mark.parametrize("backend", ["asc"])
def test_simdvf_vxor_vnot(backend):
    n = 64
    kernel = tilelang.compile(bitwise_kernel(n, backend), target=backend, out_idx=[2, 3])
    device = torch.device("npu")
    a = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device="cpu").to(device)
    b = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device="cpu").to(device)

    xor, not_ = kernel(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(xor.cpu(), torch.bitwise_xor(a.cpu(), b.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(not_.cpu(), torch.bitwise_not(a.cpu()), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
