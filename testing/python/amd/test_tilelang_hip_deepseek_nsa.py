import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_nsa"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_tilelang_nsa_decode as nsa_decode  # noqa: E402
import example_tilelang_nsa_fwd as nsa_forward  # noqa: E402
from reference import naive_nsa, naive_nsa_simple_inference  # noqa: E402


def _inputs(query_length: int):
    """Create a bounded grouped-query NSA case with one valid selected block."""
    batch, sequence_length, kv_heads, query_heads, dim = 1, 64, 1, 16, 32
    selected_blocks, block_size = 1, 32

    query = torch.randn(
        batch,
        query_length,
        query_heads,
        dim,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(batch, sequence_length, kv_heads, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    block_indices = torch.zeros(
        batch,
        query_length,
        kv_heads,
        selected_blocks,
        device="cuda",
        dtype=torch.int32,
    )
    block_counts = torch.ones(
        batch,
        query_length,
        kv_heads,
        device="cuda",
        dtype=torch.int32,
    )
    return query, key, value, block_indices, block_counts, block_size


@tilelang.testing.requires_rocm
def test_deepseek_nsa_forward():
    """Validate causal NSA forward on a wave64 launch."""
    torch.manual_seed(0)
    query, key, value, block_indices, block_counts, block_size = _inputs(query_length=64)
    scale = 0.1

    output = nsa_forward.native_sparse_attention(
        query,
        key,
        value,
        block_indices,
        dim=query.shape[-1],
        is_causal=True,
        block_size=block_size,
        groups=query.shape[2] // key.shape[2],
        selected_blocks=block_indices.shape[-1],
        scale=scale,
    )
    reference = naive_nsa(
        q=query,
        k=key,
        v=value,
        g_slc=torch.ones(query.shape[:-1], device="cuda", dtype=query.dtype),
        g_swa=torch.ones(query.shape[:-1], device="cuda", dtype=query.dtype),
        block_indices=block_indices.to(torch.long),
        block_counts=block_counts.to(torch.long),
        block_size=block_size,
        scale=scale,
    )

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_rocm
def test_deepseek_nsa_decode():
    """Validate single-token NSA decode on a wave64 launch."""
    torch.manual_seed(0)
    query, key, value, block_indices, block_counts, block_size = _inputs(query_length=1)

    output = nsa_decode.native_sparse_attention(
        query,
        key,
        value,
        block_indices,
        dim=query.shape[-1],
        block_size=block_size,
        groups=query.shape[2] // key.shape[2],
        selected_blocks=block_indices.shape[-1],
    )
    reference = naive_nsa_simple_inference(
        q=query,
        k=key,
        v=value,
        block_indices=block_indices.to(torch.long),
        block_counts=block_counts.to(torch.long),
        block_size=block_size,
    )

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
