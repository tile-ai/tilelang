import pytest
import torch

import tilelang.testing
from examples.kpool.example_glm53_kpool_compress import GLM53_HEAD_DIM
from examples.kpool.example_glm53_kpool_fp8_mqa_logits import GLM53_INDEX_HEADS
from examples.kpool.example_glm53_kpool_fused_selector import (
    glm53_kpool_fused_select,
    glm53_kpool_fused_select_reference,
    pack_glm53_kpool_cache,
)
from examples.kpool.example_glm53_kpool_topk_transform import GLM53_INDEX_TOPK
from tilelang.language.fp8 import determine_torch_fp8_type


def _make_inputs(num_rows=2, num_pools=12, page_size=8):
    device = torch.device("cuda")
    fp8_dtype = determine_torch_fp8_type(device=device)
    num_blocks = (num_pools + page_size - 1) // page_size + 2
    # Keep the fused integration fixture rank-separated. The standalone
    # scorer already checks random mixed-sign inputs with numerical
    # tolerances; an exact Top-K assertion must not depend on two logits near
    # the cutoff rounding identically in PyTorch and the MFMA accumulation.
    query_bf16 = torch.zeros(
        num_rows,
        GLM53_INDEX_HEADS,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query_bf16[:, 0, 0] = 1.0
    query = query_bf16.to(fp8_dtype)

    k_cache_bf16 = torch.zeros(
        num_blocks,
        page_size,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache_bf16[:, :, 0] = torch.arange(
        1,
        page_size + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = k_cache_bf16.to(fp8_dtype)
    block_scale = torch.pow(
        2.0,
        torch.arange(num_blocks, dtype=torch.float32, device=device),
    )[:, None]
    offset_scale = 1.0 + torch.arange(page_size, dtype=torch.float32, device=device)[None, :] / 16.0
    scale_cache = block_scale * offset_scale
    cache_u8 = pack_glm53_kpool_cache(k_cache, scale_cache)
    weights = torch.zeros(num_rows, GLM53_INDEX_HEADS, dtype=torch.float32, device=device)
    weights[:, 0] = 1.0
    pool_page_table = torch.stack(
        (
            torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32, device=device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
        )
    )
    pool_page_table_rows = torch.tensor([0, 1], dtype=torch.int32, device=device)
    pool_starts = torch.tensor([0, 2], dtype=torch.int32, device=device)
    pool_ends = torch.tensor([10, 5], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([43, 23], dtype=torch.int32, device=device)
    return (
        query,
        cache_u8,
        weights,
        pool_page_table,
        pool_page_table_rows,
        pool_starts,
        pool_ends,
        seq_lens,
    )


def _assert_history_sets_and_tail_match(actual, expected, history_lengths):
    for row, history_len in enumerate(history_lengths):
        torch.testing.assert_close(
            torch.sort(actual[row, :history_len]).values,
            torch.sort(expected[row, :history_len]).values,
        )
        assert torch.equal(actual[row, history_len:], expected[row, history_len:])


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_identity_long_and_short_rows():
    """Fuse scoring and selection while preserving long and short row semantics."""
    inputs = _make_inputs()
    expected = glm53_kpool_fused_select_reference(*inputs, token_topk=16)
    workspace = torch.empty((2, 12), dtype=torch.float32, device="cuda")
    out = torch.empty((2, 19), dtype=torch.int32, device="cuda")
    actual = glm53_kpool_fused_select(
        *inputs,
        token_topk=16,
        logits_workspace=workspace,
        out=out,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == out.data_ptr()
    _assert_history_sets_and_tail_match(actual, expected, [16, 12])

    replay_workspace = torch.empty_like(workspace)
    replay_out = torch.empty_like(out)
    replay_actual = glm53_kpool_fused_select(
        *inputs,
        token_topk=16,
        logits_workspace=replay_workspace,
        out=replay_out,
        validate=False,
    )
    torch.cuda.synchronize()
    assert replay_actual.data_ptr() == replay_out.data_ptr()
    _assert_history_sets_and_tail_match(replay_actual, expected, [16, 12])


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_bounds_work_to_active_range():
    """Start scoring and rescans at the first chunk intersecting the range."""
    inputs = list(_make_inputs(num_pools=1032, page_size=64))
    inputs[5] = torch.tensor([1025, 2], dtype=torch.int32, device="cuda")
    inputs[6] = torch.tensor([1032, 5], dtype=torch.int32, device="cuda")
    inputs[7] = torch.tensor([4131, 23], dtype=torch.int32, device="cuda")
    expected = glm53_kpool_fused_select_reference(*inputs, token_topk=16)
    workspace = torch.full((2, 1040), torch.nan, dtype=torch.float32, device="cuda")

    actual = glm53_kpool_fused_select(
        *inputs,
        token_topk=16,
        logits_workspace=workspace,
    )
    torch.cuda.synchronize()

    _assert_history_sets_and_tail_match(actual, expected, [16, 12])
    assert torch.all(torch.isnan(workspace[0, :1025]))
    assert torch.all(torch.isnan(workspace[0, 1032:]))


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_paged_and_ragged():
    """Emit framework page-table indices or ragged global offsets directly."""
    inputs = _make_inputs()
    token_page_table = torch.stack(
        (
            torch.arange(100, 148, dtype=torch.int32, device="cuda"),
            torch.arange(500, 548, dtype=torch.int32, device="cuda"),
        )
    )
    token_page_table_rows = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    expected_paged = glm53_kpool_fused_select_reference(
        *inputs,
        token_topk=16,
        token_page_table=token_page_table,
        token_page_table_rows=token_page_table_rows,
    )
    actual_paged = glm53_kpool_fused_select(
        *inputs,
        token_topk=16,
        token_page_table=token_page_table,
        token_page_table_rows=token_page_table_rows,
    )

    offsets = torch.tensor([1000, 2000], dtype=torch.int32, device="cuda")
    expected_ragged = glm53_kpool_fused_select_reference(
        *inputs,
        token_topk=16,
        topk_offsets=offsets,
    )
    actual_ragged = glm53_kpool_fused_select(
        *inputs,
        token_topk=16,
        topk_offsets=offsets,
    )
    torch.cuda.synchronize()

    _assert_history_sets_and_tail_match(actual_paged, expected_paged, [16, 12])
    _assert_history_sets_and_tail_match(actual_ragged, expected_ragged, [16, 12])


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_published_width():
    """Exercise the checkpoint's 512-pool selection and three-token tail."""
    device = torch.device("cuda")
    fp8_dtype = determine_torch_fp8_type(device=device)
    page_size = 64
    num_pools = 520
    num_blocks = (num_pools + page_size - 1) // page_size
    query = torch.full(
        (1, GLM53_INDEX_HEADS, GLM53_HEAD_DIM),
        0.125,
        dtype=torch.bfloat16,
        device=device,
    ).to(fp8_dtype)
    k_cache = torch.ones(
        num_blocks,
        page_size,
        GLM53_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(fp8_dtype)
    k_cache.view(-1, GLM53_HEAD_DIM)[:8].zero_()
    scale_cache = torch.ones(num_blocks, page_size, dtype=torch.float32, device=device)
    cache_u8 = pack_glm53_kpool_cache(k_cache, scale_cache)
    weights = torch.ones(1, GLM53_INDEX_HEADS, dtype=torch.float32, device=device)
    pool_page_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    rows = torch.zeros(1, dtype=torch.int32, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    ends = torch.full((1,), num_pools, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_pools * 4 + 3], dtype=torch.int32, device=device)

    expected = glm53_kpool_fused_select_reference(
        query,
        cache_u8,
        weights,
        pool_page_table,
        rows,
        starts,
        ends,
        seq_lens,
    )
    actual = glm53_kpool_fused_select(
        query,
        cache_u8,
        weights,
        pool_page_table,
        rows,
        starts,
        ends,
        seq_lens,
    )
    torch.cuda.synchronize()

    assert actual.shape == (1, GLM53_INDEX_TOPK + 3)
    _assert_history_sets_and_tail_match(actual, expected, [GLM53_INDEX_TOPK])


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_handles_large_equal_score_bucket():
    """Select safely when more than 4K pools have identical scores."""
    device = torch.device("cuda")
    fp8_dtype = determine_torch_fp8_type(device=device)
    page_size = 64
    num_pools = 4097
    num_blocks = (num_pools + page_size - 1) // page_size
    query = torch.zeros(
        1,
        GLM53_INDEX_HEADS,
        GLM53_HEAD_DIM,
        dtype=fp8_dtype,
        device=device,
    )
    k_cache = torch.zeros(
        num_blocks,
        page_size,
        GLM53_HEAD_DIM,
        dtype=fp8_dtype,
        device=device,
    )
    scale_cache = torch.ones(num_blocks, page_size, dtype=torch.float32, device=device)
    cache_u8 = pack_glm53_kpool_cache(k_cache, scale_cache)
    weights = torch.ones(1, GLM53_INDEX_HEADS, dtype=torch.float32, device=device)
    pool_page_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    rows = torch.zeros(1, dtype=torch.int32, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    ends = torch.full((1,), num_pools, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_pools * 4 + 2], dtype=torch.int32, device=device)
    workspace = torch.empty((1, num_pools), dtype=torch.float32, device=device)

    actual = glm53_kpool_fused_select(
        query,
        cache_u8,
        weights,
        pool_page_table,
        rows,
        starts,
        ends,
        seq_lens,
        token_topk=16,
        logits_workspace=workspace,
    )
    torch.cuda.synchronize()

    history = actual[0, :16].view(4, 4)
    groups = history[:, 0] // 4
    assert torch.equal(history, groups[:, None] * 4 + torch.arange(4, device=device))
    assert torch.unique(groups).numel() == 4
    assert torch.all((groups >= 0) & (groups < num_pools))
    assert torch.equal(
        actual[0, 16:18],
        torch.tensor([num_pools * 4, num_pools * 4 + 1], dtype=torch.int32, device=device),
    )
    assert torch.all(actual[0, 18:] == -1)
    assert torch.all(workspace[0] == 0)


@tilelang.testing.requires_rocm
def test_glm53_kpool_fused_selector_zero_history_and_validation():
    """Keep tail-only rows valid and reject undersized caller-owned scratch."""
    inputs = list(_make_inputs())
    inputs[5] = torch.zeros(2, dtype=torch.int32, device="cuda")
    inputs[6] = torch.zeros(2, dtype=torch.int32, device="cuda")
    inputs[7] = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    expected = glm53_kpool_fused_select_reference(*inputs)
    actual = glm53_kpool_fused_select(
        *inputs,
        logits_workspace=torch.empty((2, 1), dtype=torch.float32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match="workspace width"):
        glm53_kpool_fused_select(
            *_make_inputs(),
            token_topk=16,
            logits_workspace=torch.empty((2, 9), dtype=torch.float32, device="cuda"),
        )

    bad_cache = inputs[1][:, :-1].contiguous()
    with pytest.raises(ValueError, match="cache_u8 width"):
        glm53_kpool_fused_select(
            inputs[0],
            bad_cache,
            *inputs[2:],
            logits_workspace=torch.empty((2, 1), dtype=torch.float32, device="cuda"),
            validate=False,
        )


if __name__ == "__main__":
    tilelang.testing.main()
