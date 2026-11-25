import pytest
import torch

import tilelang.testing
from examples.kpool.example_glm53_kpool_topk_transform import (
    GLM53_INDEX_TOPK,
    glm53_kpool_topk_transform,
    glm53_kpool_topk_transform_reference,
)


def _assert_history_sets_and_tail_match(actual, expected, history_lengths):
    for row, history_len in enumerate(history_lengths):
        torch.testing.assert_close(
            torch.sort(actual[row, :history_len]).values,
            torch.sort(expected[row, :history_len]).values,
        )
        assert torch.equal(actual[row, history_len:], expected[row, history_len:])


@tilelang.testing.requires_rocm
def test_glm53_kpool_topk_transform_identity_long_and_short_rows():
    """Select long histories, enumerate short ones, and append compact tails."""
    torch.manual_seed(20260919)
    logits = torch.randn(2, 12, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    ends = torch.tensor([10, 5], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([43, 23], dtype=torch.int32, device="cuda")

    expected = glm53_kpool_topk_transform_reference(logits, starts, ends, seq_lens, token_topk=16)
    actual = glm53_kpool_topk_transform(logits, starts, ends, seq_lens, token_topk=16)
    torch.cuda.synchronize()

    _assert_history_sets_and_tail_match(actual, expected, [16, 12])


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("num_pools,seq_len", [(8, 21), (0, 3)])
def test_glm53_kpool_topk_transform_default_budget_short_width(num_pools, seq_len):
    """Enumerate short histories even when the published budget is wider."""
    torch.manual_seed(20260923)
    logits = torch.randn(1, num_pools, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), min(num_pools, 5), dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    expected = glm53_kpool_topk_transform_reference(logits, starts, ends, seq_lens)
    actual = glm53_kpool_topk_transform(logits, starts, ends, seq_lens)
    torch.cuda.synchronize()

    history_len = int(ends[0].item()) * 4
    _assert_history_sets_and_tail_match(actual, expected, [history_len])


@tilelang.testing.requires_rocm
def test_glm53_kpool_topk_transform_paged_and_ragged():
    """Apply request-selected token maps or ragged offsets after expansion."""
    torch.manual_seed(20260920)
    logits = torch.randn(2, 12, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    ends = torch.tensor([10, 5], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([43, 23], dtype=torch.int32, device="cuda")
    token_page_table = torch.stack(
        (
            torch.arange(100, 148, dtype=torch.int32, device="cuda"),
            torch.arange(500, 548, dtype=torch.int32, device="cuda"),
        )
    )
    page_table_rows = torch.tensor([1, 0], dtype=torch.int32, device="cuda")

    expected_paged = glm53_kpool_topk_transform_reference(
        logits,
        starts,
        ends,
        seq_lens,
        token_topk=16,
        token_page_table=token_page_table,
        page_table_rows=page_table_rows,
    )
    actual_paged = glm53_kpool_topk_transform(
        logits,
        starts,
        ends,
        seq_lens,
        token_topk=16,
        token_page_table=token_page_table,
        page_table_rows=page_table_rows,
    )
    offsets = torch.tensor([1000, 2000], dtype=torch.int32, device="cuda")
    expected_ragged = glm53_kpool_topk_transform_reference(logits, starts, ends, seq_lens, token_topk=16, topk_offsets=offsets)
    actual_ragged = glm53_kpool_topk_transform(logits, starts, ends, seq_lens, token_topk=16, topk_offsets=offsets)
    torch.cuda.synchronize()

    _assert_history_sets_and_tail_match(actual_paged, expected_paged, [16, 12])
    _assert_history_sets_and_tail_match(actual_ragged, expected_ragged, [16, 12])


@tilelang.testing.requires_rocm
def test_glm53_kpool_topk_transform_published_width():
    """Exercise the checkpoint's 512-pool to 2,048-token selection geometry."""
    torch.manual_seed(20260921)
    num_pools = 4096
    logits = torch.randn(1, num_pools, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), num_pools, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([num_pools * 4 + 3], dtype=torch.int32, device="cuda")

    expected = glm53_kpool_topk_transform_reference(logits, starts, ends, seq_lens)
    actual = glm53_kpool_topk_transform(logits, starts, ends, seq_lens)
    torch.cuda.synchronize()

    assert actual.shape == (1, GLM53_INDEX_TOPK + 3)
    _assert_history_sets_and_tail_match(actual, expected, [GLM53_INDEX_TOPK])


@tilelang.testing.requires_rocm
def test_glm53_kpool_topk_transform_rejects_unsafe_metadata():
    """Reject incompatible budgets, tail lengths, and transform metadata."""
    logits = torch.randn(1, 8, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), 5, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([21], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="positive multiple"):
        glm53_kpool_topk_transform(logits, starts, ends, seq_lens, token_topk=15)

    bad_seq_lens = torch.tensor([24], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="tail tokens"):
        glm53_kpool_topk_transform(logits, starts, ends, bad_seq_lens, token_topk=16)

    token_page_table = torch.arange(32, dtype=torch.int32, device="cuda").unsqueeze(0)
    offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="mutually exclusive"):
        glm53_kpool_topk_transform(
            logits,
            starts,
            ends,
            seq_lens,
            token_topk=16,
            token_page_table=token_page_table,
            topk_offsets=offsets,
        )


if __name__ == "__main__":
    tilelang.testing.main()
