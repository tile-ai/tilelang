import pytest
import torch

import tilelang.testing
from examples.kpool.example_glm53_kpool_compress import (
    GLM53_HEAD_DIM,
    GLM53_POOL_SIZE,
    glm53_kpool_compress_and_write_cache,
    glm53_kpool_reference,
)
from tilelang.language.fp8 import determine_torch_fp8_type


def _assert_fp8_within_one_ulp(actual, expected):
    """Allow only adjacent finite FP8 encodings across backend math paths."""
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(expected.float()).all()

    def ordered_codes(values):
        bits = values.contiguous().view(torch.uint8).to(torch.int16)
        magnitude = bits & 0x7F
        return torch.where((bits & 0x80) != 0, 0x80 - magnitude, 0x80 + magnitude)

    ulp_distance = (ordered_codes(actual) - ordered_codes(expected)).abs()
    assert torch.all(ulp_distance <= 1), f"maximum FP8 ULP distance was {int(ulp_distance.max().item())}"


def _make_inputs(num_pools=7, pool_size=GLM53_POOL_SIZE):
    """Create deterministic published-GLM-shaped tensors on the active ROCm device."""
    torch.manual_seed(20260918)
    slot_k = torch.randn(
        num_pools,
        pool_size,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    slot_score = torch.randn_like(slot_k)
    ape = torch.randn(pool_size, GLM53_HEAD_DIM, dtype=torch.float32, device="cuda")
    return slot_k, slot_score, ape


def test_glm53_kpool_published_geometry():
    """Keep the standalone kernel default aligned with the public checkpoint."""
    assert GLM53_POOL_SIZE == 4
    assert GLM53_HEAD_DIM == 128


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("round_scale", [False, True])
def test_glm53_kpool_compress_reordered_masked_cache_write(round_scale):
    """Match the reference and preserve every cache slot not actively written."""
    slot_k, slot_score, ape = _make_inputs()
    fp8_dtype = determine_torch_fp8_type(device=slot_k.device)
    num_blocks, page_size = 3, 16
    k_cache = torch.full(
        (num_blocks, page_size, GLM53_HEAD_DIM),
        1.0,
        dtype=fp8_dtype,
        device="cuda",
    )
    scale_cache = torch.full((num_blocks, page_size), -7.0, dtype=torch.float32, device="cuda")
    original_k = k_cache.clone()
    original_scale = scale_cache.clone()

    # Deliberately reorder physical locations. Masked rows use negative
    # sentinels, which must never be dereferenced by the device kernel.
    loc = torch.tensor([31, -1, 2, 40, 17, -123, 9], dtype=torch.int64, device="cuda")
    write_mask = torch.tensor([True, False, True, True, True, False, True], dtype=torch.bool, device="cuda")

    expected_k, expected_scale = glm53_kpool_reference(slot_k, slot_score, ape, round_scale=round_scale)
    glm53_kpool_compress_and_write_cache(
        slot_k,
        slot_score,
        ape,
        loc,
        k_cache,
        scale_cache,
        write_mask=write_mask,
        round_scale=round_scale,
    )
    torch.cuda.synchronize()

    flat_k = k_cache.view(-1, GLM53_HEAD_DIM)
    flat_scale = scale_cache.view(-1)
    active_rows = write_mask.nonzero(as_tuple=False).flatten()
    active_locs = loc[active_rows]
    _assert_fp8_within_one_ulp(flat_k[active_locs], expected_k[active_rows])
    torch.testing.assert_close(flat_scale[active_locs], expected_scale[active_rows], rtol=2e-3, atol=1e-6)

    untouched = torch.ones(num_blocks * page_size, dtype=torch.bool, device="cuda")
    untouched[active_locs] = False
    assert torch.equal(flat_k[untouched], original_k.view(-1, GLM53_HEAD_DIM)[untouched])
    assert torch.equal(flat_scale[untouched], original_scale.view(-1)[untouched])


@tilelang.testing.requires_rocm
def test_glm53_kpool_compress_default_mask_and_empty_input():
    """Cover the implicit all-true mask and the no-launch empty-input path."""
    slot_k, slot_score, ape = _make_inputs(num_pools=3)
    fp8_dtype = determine_torch_fp8_type(device=slot_k.device)
    k_cache = torch.zeros((1, 8, GLM53_HEAD_DIM), dtype=fp8_dtype, device="cuda")
    scale_cache = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
    loc = torch.tensor([6, 0, 4], dtype=torch.int64, device="cuda")

    expected_k, expected_scale = glm53_kpool_reference(slot_k, slot_score, ape)
    glm53_kpool_compress_and_write_cache(slot_k, slot_score, ape, loc, k_cache, scale_cache)
    torch.cuda.synchronize()
    _assert_fp8_within_one_ulp(k_cache.view(-1, GLM53_HEAD_DIM)[loc], expected_k)
    torch.testing.assert_close(scale_cache.view(-1)[loc], expected_scale, rtol=0, atol=0)

    empty_k = slot_k[:0]
    empty_score = slot_score[:0]
    before_k = k_cache.clone()
    before_scale = scale_cache.clone()
    glm53_kpool_compress_and_write_cache(
        empty_k,
        empty_score,
        ape,
        loc[:0],
        k_cache,
        scale_cache,
    )
    assert torch.equal(k_cache, before_k)
    assert torch.equal(scale_cache, before_scale)


@tilelang.testing.requires_rocm
def test_glm53_kpool_compress_rejects_unsafe_metadata():
    """Reject metadata that could cause ambiguous or out-of-bounds writes."""
    slot_k, slot_score, ape = _make_inputs(num_pools=2)
    fp8_dtype = determine_torch_fp8_type(device=slot_k.device)
    k_cache = torch.zeros((1, 4, GLM53_HEAD_DIM), dtype=fp8_dtype, device="cuda")
    scale_cache = torch.zeros((1, 4), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="unique"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape,
            torch.tensor([2, 2], dtype=torch.int64, device="cuda"),
            k_cache,
            scale_cache,
        )

    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape,
            torch.tensor([0, 4], dtype=torch.int64, device="cuda"),
            k_cache,
            scale_cache,
        )

    with pytest.raises(TypeError, match="loc must be torch.int64"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            k_cache,
            scale_cache,
        )

    with pytest.raises(ValueError, match="ape must have shape"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape[:, :-1],
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            k_cache,
            scale_cache,
        )

    with pytest.raises(ValueError, match="head_dim=128"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            k_cache,
            scale_cache,
            head_dim=64,
        )

    with pytest.raises(ValueError, match="pool_size must be positive"):
        glm53_kpool_compress_and_write_cache(
            slot_k,
            slot_score,
            ape,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            k_cache,
            scale_cache,
            pool_size=0,
        )


if __name__ == "__main__":
    tilelang.testing.main()
