"""CPU contract tests for the MXFP6 (E2M3/E3M2 + UE8M0) quantizer."""

import pytest

torch = pytest.importorskip("torch")
tilelang_testing = pytest.importorskip("tilelang.testing")

from examples.dequantize_gemm.quantize import (
    decode_fp6_values,
    decode_ue8m0_scale_bytes,
    encode_fp6_values,
    pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes,
    pack_fp6_codes,
    quantize_bf16_to_mxfp6_blockscaled,
    unpack_fp6_bytes,
)


@pytest.mark.parametrize("dtype", ["e2m3", "e3m2"])
def test_fp6_code_roundtrip_exhaustive(dtype):
    # Every one of the 64 codes must decode -> encode back to itself
    # (modulo the two zero codes: -0.0 re-encodes as +0.0).
    codes = torch.arange(64, dtype=torch.uint8)
    values = decode_fp6_values(codes, dtype)
    re = encode_fp6_values(values, dtype)
    neg_zero = 32  # sign bit set, magnitude 0
    mask = codes != neg_zero
    assert torch.equal(re[mask], codes[mask])
    assert int(re[neg_zero]) in (0, 32)


def test_fp6_pack_unpack_roundtrip():
    torch.manual_seed(0)
    codes = torch.randint(0, 64, (8, 64), dtype=torch.uint8)
    packed = pack_fp6_codes(codes)
    assert packed.shape == (8, 48)  # 0.75 bytes per element
    assert torch.equal(unpack_fp6_bytes(packed, 64), codes)


def test_fp6_pack_bit_order_matches_su6_stream():
    # Element i occupies bits [6i, 6i+6) LSB-first - the exact stream the
    # b6x16_p32 ldmatrix consumes (hardware-pinned in the sub-byte tests).
    codes = torch.zeros((1, 4), dtype=torch.uint8)
    codes[0, 0] = 0x3F
    packed = pack_fp6_codes(codes)
    assert packed[0, 0] == 0x3F and packed[0, 1] == 0 and packed[0, 2] == 0
    codes = torch.zeros((1, 4), dtype=torch.uint8)
    codes[0, 1] = 0x3F  # bits [6, 12)
    packed = pack_fp6_codes(codes)
    assert packed[0, 0] == 0xC0 and packed[0, 1] == 0x0F and packed[0, 2] == 0


@pytest.mark.parametrize("dtype", ["e2m3", "e3m2"])
def test_quantize_bf16_to_mxfp6_blockscaled_contract(dtype):
    rows, cols = 128, 256
    x = torch.randn((rows, cols), dtype=torch.bfloat16)
    blob, packed_scales, scale_bytes = quantize_bf16_to_mxfp6_blockscaled(x, dtype=dtype, return_scale_bytes=True)
    assert blob.dtype == torch.uint8 and blob.shape == (rows, cols * 3 // 4)
    assert packed_scales.dtype == torch.uint32 and packed_scales.shape == (rows, cols // 128)
    assert scale_bytes.shape == (rows, cols // 32)
    assert torch.equal(packed_scales, pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_words=2))


def test_quantize_mxfp6_compression_invariant():
    rows, cols = 128, 256
    x = torch.randn((rows, cols), dtype=torch.bfloat16)
    blob, _ = quantize_bf16_to_mxfp6_blockscaled(x)
    assert blob.element_size() * blob.numel() == rows * cols * 3 // 4


@pytest.mark.parametrize("dtype,rel_bits,sub_quantum", [("e2m3", 3, 2.0**-4), ("e3m2", 2, 2.0**-5)])
def test_quantize_bf16_to_mxfp6_blockscaled_has_bounded_error(dtype, rel_bits, sub_quantum):
    rows, cols = 128, 512
    generator = torch.Generator(device="cpu").manual_seed(11)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 3.0).to(torch.bfloat16)
    blob, _, scale_bytes = quantize_bf16_to_mxfp6_blockscaled(x, dtype=dtype, return_scale_bytes=True)
    codes = unpack_fp6_bytes(blob, cols)
    scales = decode_ue8m0_scale_bytes(scale_bytes).repeat_interleave(32, dim=1)
    reconstructed = decode_fp6_values(codes, dtype) * scales
    x_f32 = x.to(torch.float32)
    # RN: error <= 0.5 ulp -> |x| * 2^-(man_bits+1) for normals plus half the
    # subnormal quantum (times the block scale) for tiny elements.
    bound = x_f32.abs() * 2.0**-(rel_bits) + scales * sub_quantum + 1e-6
    assert bool(((x_f32 - reconstructed).abs() <= bound).all())


def test_quantize_mxfp6_nan_input_zeroes_its_block():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("nan")
    blob, _, sbytes = quantize_bf16_to_mxfp6_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0
    assert bool((unpack_fp6_bytes(blob, 256)[0, :32] == 0).all())


def test_quantize_mxfp6_inf_saturates_block_scale():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("inf")
    _, _, sbytes = quantize_bf16_to_mxfp6_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0xFE


def test_quantize_mxfp6_all_zero_block():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, :32] = 0.0
    blob, _, sbytes = quantize_bf16_to_mxfp6_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0
    assert bool((unpack_fp6_bytes(blob, 256)[0, :32] == 0).all())


if __name__ == "__main__":
    tilelang_testing.main()
