"""NPU test for T.simd.dhistv2 and T.simd.chistv2."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def histv2_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "uint8"), B: T.Tensor((4, 128), "uint16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256,), "uint8")
            b_ub = T.alloc_shared((4, 128), "uint16")
            T.copy(A, a_ub)

            with T.SimdVF():
                m8 = T.simd.pset(8)
                m16 = T.simd.pset(16)
                src = T.simd.vld(a_ub[0], dist="NORM")

                frequency0 = T.simd.alloc_var("uint16")
                frequency1 = T.simd.alloc_var("uint16")
                cumulative0 = T.simd.alloc_var("uint16")
                cumulative1 = T.simd.alloc_var("uint16")
                frequency0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                frequency1 = T.simd.vdup(T.uint16(0), "uint16", m16)
                cumulative0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                cumulative1 = T.simd.vdup(T.uint16(0), "uint16", m16)

                T.simd.dhistv2(frequency0, src, m8, bin=0)
                T.simd.dhistv2(frequency1, src, m8, bin=1)
                T.simd.chistv2(cumulative0, src, m8, bin=0)
                T.simd.chistv2(cumulative1, src, m8, bin=1)

                T.simd.vsts(b_ub[0, 0], frequency0, m16, dist="NORM_B16")
                T.simd.vsts(b_ub[1, 0], frequency1, m16, dist="NORM_B16")
                T.simd.vsts(b_ub[2, 0], cumulative0, m16, dist="NORM_B16")
                T.simd.vsts(b_ub[3, 0], cumulative1, m16, dist="NORM_B16")

            T.copy(b_ub, B)

    return main


def test_simdvf_histv2():
    device = torch.device("npu")
    source = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    output = torch.zeros((4, 128), dtype=torch.int16, device=device).view(torch.uint16)

    kernel = histv2_kernel()
    kernel(source, output)
    torch.npu.synchronize()

    frequency = torch.ones(128, dtype=torch.int16, device=device)
    cumulative0 = torch.arange(1, 129, dtype=torch.int32, device=device).to(torch.int16)
    cumulative1 = torch.arange(129, 257, dtype=torch.int32, device=device).to(torch.int16)
    signed_output = output.view(torch.int16)
    torch.testing.assert_close(signed_output[0], frequency)
    torch.testing.assert_close(signed_output[1], frequency)
    torch.testing.assert_close(signed_output[2], cumulative0)
    torch.testing.assert_close(signed_output[3], cumulative1)


def pto_histogram_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "uint8"), B: T.Tensor((2, 256), "uint16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256,), "uint8")
            b_ub = T.alloc_shared((2, 256), "uint16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                source = T.vmi.vload(a_ub[0], size=256)
                mask = T.vmi.create_mask(256, size=256)
                distribution = T.vmi.vload(b_ub[0, 0], size=256)
                cumulative = T.vmi.vload(b_ub[1, 0], size=256)
                distribution = T.vmi.vdhist(distribution, source, mask)
                cumulative = T.vmi.vchist(cumulative, source, mask)
                T.vmi.vstore(distribution, b_ub[0, 0], mask)
                T.vmi.vstore(cumulative, b_ub[1, 0], mask)
            T.copy(b_ub, B)

    return main


@pytest.mark.pto
def test_pto_histogram():
    kernel = tilelang.compile(pto_histogram_kernel(), target="pto")
    source = kernel.get_kernel_source()
    assert "pto.vmi.vdhist(" in source
    assert "pto.vmi.vchist(" in source

    device = torch.device("npu")
    values = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    output = torch.zeros((2, 256), dtype=torch.int16, device=device).view(torch.uint16)
    kernel(values, output)
    torch.npu.synchronize()

    signed = output.view(torch.int16)
    frequency = torch.ones(256, dtype=torch.int16, device=device)
    cumulative = torch.arange(1, 257, dtype=torch.int32, device=device).to(torch.int16)
    torch.testing.assert_close(signed[0], frequency)
    torch.testing.assert_close(signed[1], cumulative)


if __name__ == "__main__":
    tilelang.testing.main()
