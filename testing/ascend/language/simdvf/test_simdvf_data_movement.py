"""SIMD interleave/deinterleave lane ordering."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


@tilelang.testing.requires_ascend
def test_interleave_and_deinterleave():
    @T.prim_func
    def kernel(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), O: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            out = T.alloc_shared((4, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF():
                x = T.simd.vld(a[0])
                y = T.simd.vld(b[0])
                low, high = T.simd.vintlv(x, y)
                even, odd = T.simd.vdintlv(x, y)
                T.simd.vsts(out[0, 0], low)
                T.simd.vsts(out[1, 0], high)
                T.simd.vsts(out[2, 0], even)
                T.simd.vsts(out[3, 0], odd)
            T.copy(out, O)

    a = torch.arange(64, dtype=torch.float32)
    b = a + 128
    interleaved = torch.stack((a, b), dim=1).flatten().reshape(2, 64)
    joined = torch.cat((a, b))
    expected = torch.cat((interleaved, torch.stack((joined[::2], joined[1::2]))))
    result = tilelang.compile(kernel, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["uint8", "int16", "float32", "bfloat16"])
def test_dual_source_store_lane_order(dtype):
    torch_dtype = getattr(torch, dtype)
    lanes = 256 // torch.tensor([], dtype=torch_dtype).element_size()

    @T.prim_func
    def kernel(A: T.Tensor((2, lanes), dtype), O: T.Tensor((2 * lanes,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((2, lanes), dtype)
            dst = T.alloc_shared((2 * lanes,), dtype)
            T.copy(A, src)
            with T.SimdVF():
                first = T.simd.vld(src[0, 0])
                second = T.simd.vld(src[1, 0])
                T.simd.vst2(dst[0], first, second)
            T.copy(dst, O)

    values = (torch.arange(lanes * 2).reshape(2, lanes) % 251).to(torch_dtype)
    result = tilelang.compile(kernel, out_idx=-1)(values.npu())
    assert torch.equal(result.cpu(), values.T.contiguous().flatten())
