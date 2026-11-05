"""Packed FP4 inputs and outputs keep logical dimensions and Torch streams."""

from functools import lru_cache

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@lru_cache(None)
def _unpack(allocate):
    extent = T.dynamic("extent", "int64")
    stride = T.dynamic("stride", "int64")

    @T.prim_func
    def main(A: T.StridedTensor((2, extent), (stride, 1), "float4_e2m1fn"), B: T.Tensor((2, extent), "float8_e4m3fn")):
        if extent > 0:
            with T.Kernel(T.ceildiv(extent, 128), 2, threads=128) as (block, row):
                for lane in T.Parallel(128):
                    index = block * 128 + lane
                    if index < extent:
                        B[row, index] = T.Cast("float8_e4m3fn", A[row, index])

    return tilelang.compile(main, out_idx=1 if allocate else None, target="cuda")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("allocate", [False, True])
@pytest.mark.parametrize(
    "storage_dtype",
    [
        torch.uint8,
        torch.int8,
        pytest.param(
            getattr(torch, "float4_e2m1fn_x2", None),
            marks=pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="Torch packed FP4 dtype unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("cells", [0, 17])
def test_noncontiguous_outer_stride_sliced_storage_and_nondefault_stream(allocate, storage_dtype, cells):
    kernel = _unpack(allocate)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # The source's first row starts inside its allocation; the two visible
        # rows retain a physical stride of 24 bytes and a shorter packed width.
        backing = torch.arange(72, device="cuda", dtype=torch.uint8).reshape(3, 24)
        source = backing[1:, :cells].view(storage_dtype)
        assert source.stride() == (24, 1)
        assert source.storage_offset() == 24
        if allocate:
            output = kernel(source)
        else:
            output = torch.empty((2, cells * 2), dtype=torch.float8_e4m3fn, device="cuda")
            kernel(source, output)
    stream.synchronize()
    raw = source.view(torch.uint8).cpu().to(torch.long)
    nibbles = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(2, cells * 2)
    expected = torch.tensor(_VALUES)[nibbles].to(torch.float8_e4m3fn).view(torch.uint8)
    assert tuple(output.shape) == (2, cells * 2)
    assert torch.equal(output.cpu().view(torch.uint8), expected)


@lru_cache(None)
def _pack():
    extent = T.dynamic("extent", "int64")

    @T.prim_func
    def main(A: T.Tensor((extent,), "float16"), B: T.Tensor((extent,), "float4_e2m1fn")):
        if extent > 0:
            with T.Kernel(T.ceildiv(extent, 128), threads=128) as block:
                for lane in T.Parallel(128):
                    index = block * 128 + lane
                    if index < extent:
                        B[index] = T.Cast("float4_e2m1fn", A[index])

    return tilelang.compile(main, out_idx=1, target="cuda")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="Torch packed FP4 dtype unavailable")
def test_callee_allocated_fp4_output_has_packed_shape_and_exact_nibbles():
    source = torch.tensor(_VALUES * 16, dtype=torch.float16, device="cuda")
    output = _pack()(source)
    expected = torch.arange(16, dtype=torch.uint8).repeat(16).reshape(-1, 2)
    expected = expected[:, 0] | (expected[:, 1] << 4)
    assert output.dtype == torch.float4_e2m1fn_x2
    assert tuple(output.shape) == (source.numel() // 2,)
    assert torch.equal(output.cpu().view(torch.uint8), expected)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="Torch packed FP4 dtype unavailable")
def test_callee_allocated_fp4_output_rejects_odd_logical_extent():
    with pytest.raises(RuntimeError, match="Packed output dimension"):
        _pack()(torch.zeros(3, dtype=torch.float16, device="cuda"))
