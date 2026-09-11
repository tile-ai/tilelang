"""CPU contract tests for the MXFP4 (E2M1 + UE8M0) scale layout utilities."""

import pytest

torch = pytest.importorskip("torch")
tilelang_testing = pytest.importorskip("tilelang.testing")

from examples.dequantize_gemm.quantize import (
    decode_ue8m0_scale_bytes,
    encode_ue8m0_scale_bytes,
    pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes,
    quantize_bf16_to_mxfp4_blockscaled,
)
from examples.dequantize_gemm.quantize.nvfp4 import decode_packed_fp4_e2m1


def _expected_blockscaled_chunk_kmajor_byte_location(row: int, k32_idx: int, k128_cols: int) -> tuple[int, int, int]:
    # Same physical order as the NVFP4 packer: 128-row K-major atoms with a
    # fixed 4-group split; only the per-byte K coverage differs (32 vs 16).
    k128_word = k32_idx // 4
    byte_lane = k32_idx % 4
    row_block = row // 128
    row_in_block = row % 128
    flat_word = row_block * 128 * k128_cols + k128_word * 128 + (row_in_block % 32) * 4 + (row_in_block // 32)
    return flat_word // k128_cols, flat_word % k128_cols, byte_lane


def _packed_byte(packed, row: int, word: int, byte_lane: int) -> int:
    return (int(packed[row, word].item()) >> (8 * byte_lane)) & 0xFF


def test_encode_ue8m0_scale_bytes_known_values():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 448.0], dtype=torch.float32)
    encoded = encode_ue8m0_scale_bytes(values, rounding="ceil")
    # ceil to the next power of two: 1.5 -> 2.0, 448 -> 512.
    assert torch.equal(encoded, torch.tensor([0x00, 0x7E, 0x7F, 0x80, 0x80, 0x88], dtype=torch.uint8))
    decoded = decode_ue8m0_scale_bytes(encoded)
    torch.testing.assert_close(decoded, torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 2.0, 512.0]))
    assert torch.isnan(decode_ue8m0_scale_bytes(torch.tensor([0xFF], dtype=torch.uint8))).all()


def test_encode_ue8m0_scale_bytes_is_a_ceiling():
    values = torch.pow(2.0, torch.linspace(-20, 20, 173, dtype=torch.float32)) * 1.0000001
    decoded = decode_ue8m0_scale_bytes(encode_ue8m0_scale_bytes(values, rounding="ceil"))
    assert bool((decoded >= values).all())
    assert bool((decoded <= values * 2.0).all())


def test_encode_ue8m0_scale_bytes_saturates_at_format_max():
    # Finite values above 2**127 (and +inf) saturate to 0xFE = 2**127; 0xFF
    # stays reserved for NaN.
    values = torch.tensor([2.0**127, torch.finfo(torch.float32).max, float("inf")], dtype=torch.float32)
    encoded = encode_ue8m0_scale_bytes(values, rounding="ceil")
    assert torch.equal(encoded, torch.full((3,), 0xFE, dtype=torch.uint8))
    assert bool((decode_ue8m0_scale_bytes(encoded) == 2.0**127).all())


@pytest.mark.parametrize("block_words", [1, 2])
def test_pack_blockscaled_chunk_kmajor_ue8m0_matches_oracle(block_words):
    rows = 256
    k = 128 * block_words * 3
    k32_cols = k // 32
    generator = torch.Generator(device="cpu").manual_seed(37 + block_words)
    scale_bytes = torch.randint(0, 255, (rows, k32_cols), generator=generator, dtype=torch.uint8)

    packed = pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_words=block_words)

    assert packed.shape == (rows, k32_cols // 4)
    assert packed.dtype == torch.uint32
    k128_cols = k32_cols // 4
    for row in (0, 1, 31, 32, 127, 128, 200, 255):
        for k32_idx in range(k32_cols):
            physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, k32_idx, k128_cols)
            assert _packed_byte(packed, physical_row, physical_word, byte_lane) == int(scale_bytes[row, k32_idx])


def test_pack_blockscaled_chunk_kmajor_ue8m0_rejects_bad_shapes():
    with pytest.raises(ValueError, match="columns multiple of 8"):
        pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(torch.zeros((128, 4), dtype=torch.uint8), block_words=2)
    with pytest.raises(ValueError, match=r"block_words in \(1, 2, 4\)"):
        pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(torch.zeros((128, 12), dtype=torch.uint8), block_words=3)
    with pytest.raises(TypeError, match="torch.uint8"):
        pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(torch.zeros((128, 8), dtype=torch.int32))


def test_quantize_bf16_to_mxfp4_blockscaled_contract():
    rows, cols = 128, 256
    x = torch.randn((rows, cols), dtype=torch.bfloat16)

    packed_fp4, packed_scales, scale_bytes = quantize_bf16_to_mxfp4_blockscaled(x, return_scale_bytes=True)

    assert packed_fp4.dtype == torch.int8 and packed_fp4.shape == (rows, cols // 2)
    assert packed_scales.dtype == torch.uint32 and packed_scales.shape == (rows, cols // 128)
    assert scale_bytes.dtype == torch.uint8 and scale_bytes.shape == (rows, cols // 32)
    assert torch.equal(
        packed_scales,
        pack_blockscaled_chunk_kmajor_ue8m0_scale_bytes(scale_bytes, block_words=2),
    )


def test_quantize_mxfp4_nan_input_zeroes_its_block():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("nan")
    packed, _, sbytes = quantize_bf16_to_mxfp4_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0
    block = decode_packed_fp4_e2m1(packed, 256)[0, :32]
    assert bool((block == 0).all())


def test_quantize_mxfp4_inf_saturates_block_scale():
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    x[0, 5] = float("inf")
    _, _, sbytes = quantize_bf16_to_mxfp4_blockscaled(x, return_scale_bytes=True)
    assert int(sbytes[0, 0]) == 0xFE


def test_quantize_bf16_to_mxfp4_blockscaled_has_bounded_error():
    rows, cols = 128, 512
    generator = torch.Generator(device="cpu").manual_seed(11)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 3.0).to(torch.bfloat16)

    packed_fp4, _, scale_bytes = quantize_bf16_to_mxfp4_blockscaled(x, return_scale_bytes=True)

    scales = decode_ue8m0_scale_bytes(scale_bytes).repeat_interleave(32, dim=1)
    reconstructed = decode_packed_fp4_e2m1(packed_fp4, cols) * scales
    x_f32 = x.to(torch.float32)
    # The ceil-power-of-two scale keeps |x| / scale within the FP4 range; the
    # worst-case FP4 rounding gap is one unit at the top of the range.
    assert bool(((x_f32 - reconstructed).abs() <= scales * 1.0 + 1e-6).all())
    zero_blocks = x_f32.reshape(rows, cols // 32, 32).abs().amax(dim=2) == 0
    assert bool((scale_bytes[zero_blocks] == 0).all()) if zero_blocks.any() else True


if __name__ == "__main__":
    tilelang_testing.main()
