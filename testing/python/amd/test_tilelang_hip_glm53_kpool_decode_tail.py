import pytest
import torch

import tilelang.testing
from examples.kpool.example_glm53_kpool_compress import GLM53_HEAD_DIM, GLM53_POOL_SIZE
from examples.kpool.example_glm53_kpool_decode_tail import (
    glm53_kpool_decode_tail_reference,
    glm53_kpool_decode_update_and_maybe_write_cache,
    glm53_kpool_seed_tail_cache,
    glm53_kpool_seed_tail_reference,
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


def _tail_slots(block, positions):
    positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
    return torch.where(
        positions >= 0,
        block * GLM53_POOL_SIZE + positions.remainder(GLM53_POOL_SIZE),
        -1,
    )


@tilelang.testing.requires_rocm
def test_glm53_kpool_seed_tail_cache_multi_request():
    """Seed only each request's final rolling window from flattened prefill."""
    torch.manual_seed(20260919)
    request_lengths = (7, 3)
    num_tokens = sum(request_lengths)
    key = torch.randn(num_tokens, GLM53_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_score = torch.randn_like(key)
    tail_slot_mapping = torch.cat(
        (
            _tail_slots(2, range(request_lengths[0])),
            _tail_slots(0, range(request_lengths[1])),
        )
    )
    cu_seqlens = torch.tensor([0, request_lengths[0], num_tokens], dtype=torch.int32, device="cuda")
    actual_tail = torch.full(
        (4, 2, GLM53_POOL_SIZE, GLM53_HEAD_DIM),
        -7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_tail = actual_tail.clone()

    glm53_kpool_seed_tail_reference(
        expected_tail,
        key,
        slot_score,
        tail_slot_mapping,
        cu_seqlens,
    )
    glm53_kpool_seed_tail_cache(
        actual_tail,
        key,
        slot_score,
        tail_slot_mapping,
        cu_seqlens,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_tail, expected_tail)
    # Request 0 has seven tokens, so only positions 3..6 form its final window.
    assert torch.equal(actual_tail[2, 0, 3], key[3])
    assert torch.equal(actual_tail[2, 0, 2], key[6])
    # Request 1 is shorter than one pool and therefore seeds every token.
    assert torch.equal(actual_tail[0, 0, 2], key[9])
    assert torch.all(actual_tail[0, 0, 3] == -7)

    shared_block_slots = torch.cat((_tail_slots(1, range(3)), _tail_slots(1, range(3))))
    shared_boundaries = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="distinct tail blocks"):
        glm53_kpool_seed_tail_cache(
            actual_tail,
            key[:6],
            slot_score[:6],
            shared_block_slots,
            shared_boundaries,
        )


@tilelang.testing.requires_rocm
def test_glm53_kpool_seed_tail_cache_rejects_nonzero_initial_phase():
    """Reject a prefill mapping that cannot represent request-local position zero."""
    key = torch.randn(3, GLM53_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_score = torch.randn_like(key)
    tail_slot_mapping = _tail_slots(0, range(5, 8))
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    tail_cache = torch.zeros(
        (1, 2, GLM53_POOL_SIZE, GLM53_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )

    with pytest.raises(ValueError, match="phase zero"):
        glm53_kpool_seed_tail_cache(
            tail_cache,
            key,
            slot_score,
            tail_slot_mapping,
            cu_seqlens,
        )


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("round_scale", [False, True])
def test_glm53_kpool_decode_tail_speculative_order_and_pool_close(round_scale):
    """Process grouped speculative tokens in order and write completed pools."""
    torch.manual_seed(20260920)
    num_requests, next_n = 3, 5
    key = torch.randn(
        num_requests,
        next_n,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    slot_score = torch.randn_like(key)
    ape = torch.randn(
        GLM53_POOL_SIZE,
        GLM53_HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    positions = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [5, 6, 7, -1, -1],
            [8, 9, 10, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    tail_slot_mapping = torch.stack(
        (
            _tail_slots(2, positions[0].tolist()),
            _tail_slots(0, positions[1].tolist()),
            _tail_slots(4, positions[2].tolist()),
        )
    )
    cache_loc = torch.full_like(positions, -1)
    cache_loc[0, 2] = 31
    cache_loc[1, 2] = 2

    initial_tail = torch.randn(
        5,
        2,
        GLM53_POOL_SIZE,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual_tail = initial_tail.clone()
    expected_tail = initial_tail.clone()
    fp8_dtype = determine_torch_fp8_type(device=key.device)
    actual_k_cache = torch.full(
        (3, 16, GLM53_HEAD_DIM),
        1.0,
        dtype=fp8_dtype,
        device="cuda",
    )
    actual_scale_cache = torch.full((3, 16), -7.0, dtype=torch.float32, device="cuda")
    expected_k_cache = actual_k_cache.clone()
    expected_scale_cache = actual_scale_cache.clone()

    glm53_kpool_decode_tail_reference(
        expected_tail,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        expected_k_cache,
        expected_scale_cache,
        round_scale=round_scale,
    )
    glm53_kpool_decode_update_and_maybe_write_cache(
        actual_tail,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        actual_k_cache,
        actual_scale_cache,
        round_scale=round_scale,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_tail, expected_tail)
    active_locs = cache_loc[cache_loc >= 0]
    _assert_fp8_within_one_ulp(
        actual_k_cache.view(-1, GLM53_HEAD_DIM)[active_locs],
        expected_k_cache.view(-1, GLM53_HEAD_DIM)[active_locs],
    )
    torch.testing.assert_close(
        actual_scale_cache.view(-1)[active_locs],
        expected_scale_cache.view(-1)[active_locs],
        rtol=2e-3 if not round_scale else 0,
        atol=1e-6 if not round_scale else 0,
    )

    untouched = torch.ones(actual_k_cache.shape[0] * actual_k_cache.shape[1], dtype=torch.bool, device="cuda")
    untouched[active_locs] = False
    assert torch.equal(
        actual_k_cache.view(-1, GLM53_HEAD_DIM)[untouched],
        expected_k_cache.view(-1, GLM53_HEAD_DIM)[untouched],
    )
    assert torch.equal(
        actual_scale_cache.view(-1)[untouched],
        expected_scale_cache.view(-1)[untouched],
    )


@tilelang.testing.requires_rocm
def test_glm53_kpool_decode_tail_two_pool_closures_reuse_shared_memory():
    """Keep two pool writes correct when one request reuses shared memory."""
    torch.manual_seed(20260922)
    positions = torch.arange(2, 8, dtype=torch.int32, device="cuda").unsqueeze(0)
    next_n = positions.shape[1]
    key = torch.randn(1, next_n, GLM53_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_score = torch.randn_like(key)
    ape = torch.randn(
        GLM53_POOL_SIZE,
        GLM53_HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    tail_slot_mapping = _tail_slots(1, positions[0].tolist()).unsqueeze(0)
    cache_loc = torch.full_like(positions, -1)
    cache_loc[0, 1] = 3
    cache_loc[0, -1] = 11

    initial_tail = torch.randn(
        2,
        2,
        GLM53_POOL_SIZE,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual_tail = initial_tail.clone()
    expected_tail = initial_tail.clone()
    fp8_dtype = determine_torch_fp8_type(device=key.device)
    actual_k_cache = torch.zeros((1, 16, GLM53_HEAD_DIM), dtype=fp8_dtype, device="cuda")
    actual_scale_cache = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    expected_k_cache = actual_k_cache.clone()
    expected_scale_cache = actual_scale_cache.clone()

    glm53_kpool_decode_tail_reference(
        expected_tail,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        expected_k_cache,
        expected_scale_cache,
    )
    glm53_kpool_decode_update_and_maybe_write_cache(
        actual_tail,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        cache_loc,
        positions,
        actual_k_cache,
        actual_scale_cache,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_tail, expected_tail)
    active_locs = cache_loc[cache_loc >= 0]
    _assert_fp8_within_one_ulp(
        actual_k_cache.view(-1, GLM53_HEAD_DIM)[active_locs],
        expected_k_cache.view(-1, GLM53_HEAD_DIM)[active_locs],
    )
    torch.testing.assert_close(
        actual_scale_cache.view(-1)[active_locs],
        expected_scale_cache.view(-1)[active_locs],
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_rocm
def test_glm53_kpool_decode_tail_rejects_unsafe_metadata():
    """Reject ordering, ownership, phase, and cache-write violations."""
    torch.manual_seed(20260921)
    key = torch.randn(2, 3, GLM53_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_score = torch.randn_like(key)
    ape = torch.randn(
        GLM53_POOL_SIZE,
        GLM53_HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    positions = torch.tensor([[13, 14, 15], [30, 31, -1]], dtype=torch.int32, device="cuda")
    tail_slot_mapping = torch.stack(
        (
            _tail_slots(0, positions[0].tolist()),
            _tail_slots(1, positions[1].tolist()),
        )
    )
    cache_loc = torch.tensor([[-1, -1, 1], [-1, 2, -1]], dtype=torch.int32, device="cuda")
    tail_cache = torch.zeros(
        (2, 2, GLM53_POOL_SIZE, GLM53_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fp8_dtype = determine_torch_fp8_type(device=key.device)
    k_cache = torch.zeros((1, 4, GLM53_HEAD_DIM), dtype=fp8_dtype, device="cuda")
    scale_cache = torch.zeros((1, 4), dtype=torch.float32, device="cuda")

    def launch(test_tail_slots=tail_slot_mapping, test_positions=positions, test_cache_loc=cache_loc):
        glm53_kpool_decode_update_and_maybe_write_cache(
            tail_cache,
            test_tail_slots,
            key,
            slot_score,
            ape,
            test_cache_loc,
            test_positions,
            k_cache,
            scale_cache,
        )

    bad_positions = positions.clone()
    bad_positions[0, 1] = 16
    with pytest.raises(ValueError, match="consecutive"):
        launch(test_positions=bad_positions)

    bad_phase = tail_slot_mapping.clone()
    bad_phase[0, 0] += 1
    with pytest.raises(ValueError, match="phase"):
        launch(test_tail_slots=bad_phase)

    shared_block = tail_slot_mapping.clone()
    shared_block[1, :2] -= GLM53_POOL_SIZE
    with pytest.raises(ValueError, match="distinct tail blocks"):
        launch(test_tail_slots=shared_block)

    missing_write = cache_loc.clone()
    missing_write[0, 2] = -1
    with pytest.raises(ValueError, match="exactly when"):
        launch(test_cache_loc=missing_write)

    duplicate_write = cache_loc.clone()
    duplicate_write[1, 1] = duplicate_write[0, 2]
    with pytest.raises(ValueError, match="unique"):
        launch(test_cache_loc=duplicate_write)


if __name__ == "__main__":
    tilelang.testing.main()
