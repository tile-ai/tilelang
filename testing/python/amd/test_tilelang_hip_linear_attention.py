import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import tilelang
import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "linear_attention"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_linear_attn_fwd as linear_attention  # noqa: E402


@tilelang.testing.requires_rocm
def test_fused_chunk_linear_attention_forward():
    """Validate output and final recurrent state on a bounded gfx942 shape."""
    torch.manual_seed(0)
    batch, seq_len, heads, dim = 1, 128, 2, 64

    query = F.normalize(
        torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.float16)
    key = F.normalize(
        torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.float16)
    value = torch.randn_like(key)

    kernel = linear_attention.tl_fused_chunk_fwd_kernel(batch, seq_len, heads, dim, dim)
    output = torch.zeros(
        batch,
        seq_len,
        heads,
        dim,
        device="cuda",
        dtype=torch.float32,
    )
    final_state = kernel(query, key, value, output)
    output_ref, final_state_ref = linear_attention.ref_program(query, key, value)

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(final_state, final_state_ref, rtol=1e-2, atol=1e-2)
