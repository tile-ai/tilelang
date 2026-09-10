"""Test for T.simd.vcpadd on Ascend SimdVF."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def vcpadd_kernel(backend="asc"):

    @T.prim_func
    def main(A: T.Buffer((64,), "float32"), B: T.Buffer((32,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((32,), "float32")

            T.copy(A, a_ub)
            with T.SimdVF():
                if backend == "pto":
                    # Emulate Ascend vcpadd: pairwise even/odd lane sums into 32 results.
                    full = T.vmi.create_mask(64, size=64)
                    src = T.vmi.vload(a_ub[0], size=64)
                    zeros = T.vmi.vbrc(T.float32(0), size=64)
                    even, odd = T.vmi.vdintlv(src, zeros, full)
                    T.vmi.vstore(
                        T.vmi.vadd(even, odd, full),
                        b_ub[0],
                        T.vmi.create_mask(32, size=64),
                    )
                else:
                    low_half = T.simd.pset(32, "PAT_VL32")
                    src = T.simd.vld(a_ub[0])
                    result = T.simd.vcpadd(src)
                    T.simd.vsts(b_ub[0], result, low_half, extent=32)
            T.copy(b_ub, B)

    return main


@pytest.mark.parametrize("backend", ["asc", pytest.param("pto", marks=pytest.mark.pto)])
def test_simdvf_vcpadd(backend):
    kernel = tilelang.compile(vcpadd_kernel(backend), target=backend, out_idx=-1)
    a = torch.arange(64, dtype=torch.float32, device="cpu").to("npu")

    result = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), a.cpu().reshape(32, 2).sum(dim=1), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
