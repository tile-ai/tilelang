import math
import sys
from pathlib import Path

import pytest
import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "seer_attention"
sys.path.insert(0, str(_EXAMPLE_DIR))

import block_sparse_attn_tilelang as seer  # noqa: E402


def _reference(query, key, value, block_mask, block_size):
    query_length = query.shape[-2]
    kv_length = key.shape[-2]
    past_length = kv_length - query_length

    full_mask = torch.kron(
        block_mask.float(),
        torch.ones(block_size, block_size, device=query.device),
    ).bool()
    full_mask = full_mask[..., :kv_length, :kv_length]
    full_mask = full_mask[..., past_length : past_length + query_length, :]

    query_positions = torch.arange(past_length, past_length + query_length, device=query.device).unsqueeze(1)
    key_positions = torch.arange(kv_length, device=query.device).unsqueeze(0)
    full_mask &= key_positions <= query_positions

    scores = torch.einsum("bhsd,bhtd->bhst", query, key) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~full_mask, float("-inf"))
    return torch.einsum("bhst,bhtd->bhsd", torch.softmax(scores, dim=-1), value)


@pytest.mark.parametrize(
    ("heads", "query_length", "kv_length", "topk"),
    ((2, 128, 128, 2), (1, 128, 256, 1)),
    ids=("self-attention", "query-shorter-than-kv"),
)
@tilelang.testing.requires_rocm
def test_seer_block_sparse_attention(heads, query_length, kv_length, topk):
    """Cover causal self-attention and decode-style Q/KV length mismatch."""
    torch.manual_seed(0)
    batch, head_dim, block_size = 1, 64, 64
    downsample_length = math.ceil(kv_length / block_size)

    query = torch.randn(batch, heads, query_length, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, heads, kv_length, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)

    mask_scores = torch.randn(
        batch,
        heads,
        downsample_length,
        downsample_length,
        device="cuda",
        dtype=torch.float16,
    )
    mask_scores[..., 0] = 100
    block_mask = seer.get_sparse_attn_mask_from_topk(mask_scores, topk=topk)

    kernel = seer.blocksparse_flashattn(
        batch,
        heads,
        query_length,
        kv_length,
        head_dim,
        downsample_length,
        is_causal=True,
    )
    output = kernel(query, key, value, block_mask.to(torch.int8))
    reference = _reference(query, key, value, block_mask, block_size)

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
