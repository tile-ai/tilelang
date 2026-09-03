import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v32"
sys.path.insert(0, str(_EXAMPLE_DIR))


@tilelang.testing.requires_rocm
def test_deepseek_v32_sparse_mla_forward():
    """Validate the portable sparse MLA forward kernel on CDNA."""
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
    )
    key_value = torch.randn(
        batch,
        sequence_length,
        kv_heads,
        query_key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = (
        torch.arange(sequence_length, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 1, sequence_length)
        .expand(batch, sequence_length, kv_heads, topk)
        .contiguous()
    )

    output, _ = sparse_mla_fwd.sparse_mla_fwd_interface(
        query,
        key_value,
        indices,
        d_v=value_dim,
        block_I=32,
        num_stages=2,
        threads=64,
    )
    reference = sparse_mla_fwd.ref_sparse_mla_fwd_interface(query, key_value, indices)

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
