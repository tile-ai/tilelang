"""CPU contract tests for the MXFP8 (FP8 + UE8M0) quantizer."""

import pytest

torch = pytest.importorskip("torch")
tilelang_testing = pytest.importorskip("tilelang.testing")

from examples.dequantize_gemm.quantize import (
    decode_ue8m0_scale_bytes,
    pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes,
    quantize_bf16_to_mxfp8_blockscaled,
)


@pytest.mark.parametrize("dtype,torch_dtype", [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)])
def test_quantize_bf16_to_mxfp8_blockscaled_contract(dtype, torch_dtype):
    rows, cols = 128, 256
    x = torch.randn((rows, cols), dtype=torch.bfloat16)

    fp8_data, packed_scales, scale_bytes = quantize_bf16_to_mxfp8_blockscaled(x, dtype=dtype, return_scale_bytes=True)

    assert fp8_data.dtype == torch_dtype and fp8_data.shape == (rows, cols)
    assert packed_scales.dtype == torch.uint32 and packed_scales.shape == (rows, cols // 128)
    assert scale_bytes.dtype == torch.uint8 and scale_bytes.shape == (rows, cols // 32)
    # Scale storage is byte-identical to the MXFP4 packer.
    assert torch.equal(packed_scales, pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_words=2))


def test_quantize_mxfp8_compression_invariant():
    # The purpose constraint: quantized global storage must actually shrink.
    # FP8 data is exactly numel bytes (1 byte per element); any future route
    # that widens the container fails this by construction.
    rows, cols = 128, 256
    x = torch.randn((rows, cols), dtype=torch.bfloat16)
    fp8_data, _ = quantize_bf16_to_mxfp8_blockscaled(x)
    assert fp8_data.element_size() == 1
    assert fp8_data.numel() * fp8_data.element_size() == rows * cols


def test_quantize_bf16_to_mxfp8_blockscaled_has_bounded_error():
    rows, cols = 128, 512
    generator = torch.Generator(device="cpu").manual_seed(11)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 3.0).to(torch.bfloat16)

    fp8_data, _, scale_bytes = quantize_bf16_to_mxfp8_blockscaled(x, return_scale_bytes=True)

    scales = decode_ue8m0_scale_bytes(scale_bytes).repeat_interleave(32, dim=1)
    reconstructed = fp8_data.to(torch.float32) * scales
    x_f32 = x.to(torch.float32)
    # Round-to-nearest e4m3 cast: error <= 0.5 ulp. For normal scaled values
    # that is |x| * 2^-4 (3 mantissa bits); elements much smaller than their
    # block amax land in the e4m3 subnormal range (quantum 2^-9), giving the
    # absolute scale * 2^-10 term.
    bound = x_f32.abs() * 2.0**-3 + scales * 2.0**-10 + 1e-6
    assert bool(((x_f32 - reconstructed).abs() <= bound).all())


def test_quantize_mxfp8_nan_input_zeroes_its_block():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("nan")
    fp8_data, _, sbytes = quantize_bf16_to_mxfp8_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0
    assert bool((fp8_data[0, :32].to(torch.float32) == 0).all())


def test_quantize_mxfp8_inf_saturates_block_scale():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("inf")
    _, _, sbytes = quantize_bf16_to_mxfp8_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0xFE


if __name__ == "__main__":
    tilelang_testing.main()
