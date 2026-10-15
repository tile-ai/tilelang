import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "attention_sink"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_gqa_sink_bwd_bhsd as gqa  # noqa: E402
import example_mha_sink_bwd_bhsd as mha  # noqa: E402


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _assert_forward_backward(module, *, heads, groups=1):
    torch.manual_seed(0)
    batch, seq_len, dim = 1, 128, 64
    kv_heads = heads // groups

    query = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, kv_heads, seq_len, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float16)
    grad = torch.randn_like(query)

    q_tl, k_tl, v_tl, sinks_tl = (_clone_with_grad(t) for t in (query, key, value, sinks))
    q_ref, k_ref, v_ref, sinks_ref = (_clone_with_grad(t) for t in (query, key, value, sinks))

    if groups == 1:
        output = module.attention(q_tl, k_tl, v_tl, sinks_tl, None)
    else:
        output = module.attention(q_tl, k_tl, v_tl, sinks_tl, None, groups)
    reference = module.ref_program(q_ref, k_ref, v_ref, sinks_ref, dtype=torch.float16)

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    output.backward(grad)
    reference.backward(grad)
    for actual, expected in (
        (q_tl.grad, q_ref.grad),
        (k_tl.grad, k_ref.grad),
        (v_tl.grad, v_ref.grad),
        (sinks_tl.grad, sinks_ref.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_rocm
def test_mha_attention_sink_forward_backward():
    _assert_forward_backward(mha, heads=1)


@tilelang.testing.requires_rocm
def test_gqa_attention_sink_forward_backward():
    _assert_forward_backward(gqa, heads=4, groups=2)
