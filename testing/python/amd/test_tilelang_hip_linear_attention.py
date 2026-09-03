import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import tilelang
import tilelang.testing
from tilelang.carver.arch import get_arch


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "linear_attention"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_linear_attn_fwd as linear_attention  # noqa: E402
import example_linear_attn_bwd as linear_attention_bwd  # noqa: E402


def _assert_launchable(kernel):
    """Report generated HIP resource overflows before the runtime launch."""
    arch = get_arch(kernel.target)
    dynamic_smem = max(
        (int(size) for size in kernel.adapter.dynamic_smem_buf.values() if size is not None),
        default=0,
    )
    assert dynamic_smem <= arch.smem_cap, f"dynamic shared memory {dynamic_smem} exceeds device limit {arch.smem_cap}"

    for name, usage in kernel.resource_usage.items():
        block_size = 1
        for extent in kernel.adapter.block_info[name]:
            block_size *= int(extent)
        assert not usage.n_regs or usage.n_regs * block_size < arch.reg_cap, (
            f"{name} uses {usage.n_regs} VGPRs per thread at the device register-file limit"
        )


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
    _assert_launchable(kernel)
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


@tilelang.testing.requires_rocm
def test_fused_chunk_linear_attention_backward():
    """Validate query, key, and value gradients on a bounded gfx942 shape."""
    torch.manual_seed(0)
    batch, seq_len, heads, dim = 1, 128, 1, 64

    query = F.normalize(
        torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.float16)
    key = F.normalize(
        torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.float16)
    value = torch.randn_like(key)
    grad_output = torch.randn_like(value)

    kernel = linear_attention_bwd.tl_fused_chunk_bwd_kernel(batch, seq_len, heads, dim, dim)
    _assert_launchable(kernel)
    actual = tuple(torch.zeros_like(query, dtype=torch.float32) for _ in range(3))
    kernel(query, key, value, grad_output, *actual)
    actual = tuple(grad.to(torch.float16) for grad in actual)

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    output_ref, _ = linear_attention_bwd.ref_program(query_ref, key_ref, value_ref)
    output_ref.backward(grad_output)
    expected = (query_ref.grad, key_ref.grad, value_ref.grad)

    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-2, atol=1e-2)
