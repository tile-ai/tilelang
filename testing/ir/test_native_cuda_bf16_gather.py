"""CUDA scalarized 16-bit gather/scatter keeps payloads and lane order."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("stride", [2, 3])
def test_gather_scatter_preserves_raw_bits(dtype, lanes, stride, tmp_path):
    size = 32 * lanes * stride + 2

    @T.prim_func
    def main(A: T.Tensor((size,), dtype), C: T.Tensor((size,), dtype)):
        with T.Kernel(1, threads=32):
            index = T.Ramp(1 + T.get_thread_binding() * lanes * stride, stride, lanes)
            C[index] = A[index]

    # Include both signed zeros, subnormals, infinities and distinct NaN payloads.
    bits = [0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80, 0x7C00, 0xFC00, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7C01, 0xFC03, 0x1234, 0x4321]
    values = [(bits[i % len(bits)] + 0x8000) % 0x10000 - 0x8000 for i in range(size)]
    raw = torch.tensor(values, dtype=torch.int16, device="cuda")
    output_bits = torch.full_like(raw, 0x5A5A)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    kernel = tilelang.compile(main)
    source = kernel.get_kernel_source()
    (tmp_path / "device.cu").write_text(source)
    assert ("__pack_nv_bfloat162" if dtype == "bfloat16" else "__pack_half2") in source
    kernel(raw.view(torch_dtype), output_bits.view(torch_dtype))
    expected = torch.full_like(raw, 0x5A5A)
    expected[1 : 1 + 32 * lanes * stride : stride] = raw[1 : 1 + 32 * lanes * stride : stride]
    torch.testing.assert_close(output_bits, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_odd_16bit_vector_lane_count_rejected(dtype):
    @T.prim_func
    def main(A: T.Tensor((192,), dtype), C: T.Tensor((192,), dtype)):
        with T.Kernel(1, threads=32):
            index = T.Ramp(T.get_thread_binding() * 6, 2, 3)
            C[index] = A[index]

    with pytest.raises(RuntimeError, match="only support even lane"):
        tilelang.compile(main)
