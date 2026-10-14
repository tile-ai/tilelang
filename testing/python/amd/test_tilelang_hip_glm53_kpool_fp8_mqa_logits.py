import pytest
import torch

import tilelang.testing
from examples.kpool.example_glm53_kpool_compress import GLM53_HEAD_DIM
from examples.kpool.example_glm53_kpool_fp8_mqa_logits import (
    GLM53_INDEX_HEADS,
    glm53_kpool_fp8_mqa_logits,
    glm53_kpool_fp8_mqa_logits_reference,
)
from tilelang.language.fp8 import determine_torch_fp8_type


def _make_inputs():
    torch.manual_seed(20260919)
    device = torch.device("cuda")
    fp8_dtype = determine_torch_fp8_type(device=device)
    num_rows, num_blocks, page_size = 3, 5, 8
    query = (
        torch.randn(
            num_rows,
            GLM53_INDEX_HEADS,
            GLM53_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).to(fp8_dtype)
    k_cache = (
        torch.randn(
            num_blocks,
            page_size,
            GLM53_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).to(fp8_dtype)
    scale_cache = torch.rand(num_blocks, page_size, dtype=torch.float32, device=device) * 0.2 + 0.05
    weights = torch.randn(num_rows, GLM53_INDEX_HEADS, dtype=torch.float32, device=device) * 0.25
    pool_page_table = torch.tensor(
        [[2, 0, 4], [1, 3, 0]],
        dtype=torch.int32,
        device=device,
    )
    page_table_rows = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    pool_starts = torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
    pool_ends = torch.tensor([17, 11, 7], dtype=torch.int32, device=device)
    return (
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
    )


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("block_n", [32, 64])
def test_glm53_kpool_fp8_mqa_logits_paged_ragged_bounds(block_n):
    """Match gated pooled logits through reordered page tables and ragged bounds."""
    inputs = _make_inputs()
    expected = glm53_kpool_fp8_mqa_logits_reference(*inputs, max_num_pools=18)
    actual = glm53_kpool_fp8_mqa_logits(*inputs, max_num_pools=18, block_n=block_n)
    torch.cuda.synchronize()

    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=2e-2, atol=2e-2)
    assert torch.isneginf(actual[~finite]).all()


@tilelang.testing.requires_rocm
def test_glm53_kpool_fp8_mqa_logits_infers_width_and_handles_empty_rows():
    """Infer output width from pool ends while preserving empty ranges."""
    inputs = _make_inputs()
    expected = glm53_kpool_fp8_mqa_logits_reference(*inputs)
    actual = glm53_kpool_fp8_mqa_logits(*inputs)
    torch.cuda.synchronize()

    assert actual.shape == (3, 17)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=2e-2, atol=2e-2)
    assert torch.isneginf(actual[2]).all()


@tilelang.testing.requires_rocm
def test_glm53_kpool_fp8_mqa_logits_rejects_unsafe_metadata():
    """Reject invalid ranges, page rows, and active physical cache blocks."""
    inputs = list(_make_inputs())

    bad_ends = inputs[7].clone()
    bad_ends[0] = 25
    with pytest.raises(ValueError, match="max_num_pools"):
        glm53_kpool_fp8_mqa_logits(*inputs[:7], bad_ends, max_num_pools=18)

    bad_rows = inputs[5].clone()
    bad_rows[1] = 2
    with pytest.raises(ValueError, match="page_table_rows"):
        glm53_kpool_fp8_mqa_logits(*inputs[:5], bad_rows, *inputs[6:], max_num_pools=18)

    bad_table = inputs[4].clone()
    bad_table[0, 1] = inputs[1].shape[0]
    with pytest.raises(ValueError, match="valid cache blocks"):
        glm53_kpool_fp8_mqa_logits(*inputs[:4], bad_table, *inputs[5:], max_num_pools=18)


if __name__ == "__main__":
    tilelang.testing.main()
