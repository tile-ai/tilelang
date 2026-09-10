import sys
from pathlib import Path

import pytest
import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v32"
sys.path.insert(0, str(_EXAMPLE_DIR))


@tilelang.testing.requires_rocm
def test_deepseek_v32_sparse_mla_backward():
    """Validate sparse MLA gradients, including colliding dKV atomics, on CDNA."""
    import sparse_mla_bwd
    import sparse_mla_fwd

    torch.manual_seed(0)
    batch, sequence_length, heads, kv_heads = 1, 32, 16, 1
    query_key_dim, value_dim, topk = 576, 512, 32

    query = torch.randn(
        batch,
        sequence_length,
        heads,
        query_key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key_value = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        query_key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output_gradient = torch.randn(
        batch,
        sequence_length,
        heads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Every query visits the same KV positions. The causal mask makes the prefix
    # valid while still forcing different query blocks to accumulate into the
    # same dKV elements through T.atomic_addx4.
    indices = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 1, topk)
        .expand(batch, sequence_length, kv_heads, topk)
        .contiguous()
    )

    output, logsumexp = sparse_mla_fwd.sparse_mla_fwd_interface(
        query,
        key_value,
        indices,
        d_v=value_dim,
        block_I=32,
        num_stages=2,
        threads=64,
    )
    actual_dq, actual_dkv = sparse_mla_bwd.sparse_mla_bwd(
        query,
        key_value,
        output,
        output_gradient,
        indices,
        logsumexp,
        block_size=16,
        threads=64,
    )
    expected_dq, expected_dkv = sparse_mla_bwd.ref_sparse_mla_bwd_interface(
        query,
        key_value,
        output,
        output_gradient,
        indices,
        logsumexp,
    )

    sparse_mla_bwd.assert_tensors_similar(actual_dq, expected_dq, eps=1e-4, name="dq")
    sparse_mla_bwd.assert_tensors_similar(actual_dkv, expected_dkv, eps=1e-4, name="dkv")

    with pytest.raises(ValueError, match="positive even integer"):
        sparse_mla_bwd.sparse_mla_bwd(
            query,
            key_value,
            output,
            output_gradient,
            indices,
            logsumexp,
            block_size=15,
            threads=64,
        )
