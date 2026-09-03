import sys
from math import prod
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
    resource_usage = kernel.resource_usage

    for global_var, func in kernel.artifact.device_mod.functions.items():
        name = global_var.name_hint
        attrs = func.attrs
        dynamic_smem = int(attrs["dyn_shared_memory_buf"]) if "dyn_shared_memory_buf" in attrs else 0
        thread_extent = attrs.get("thread_extent", {})
        block_size = prod(int(extent) for tag, extent in thread_extent.items() if str(tag).startswith("threadIdx"))
        usage = resource_usage.get(name)

        assert dynamic_smem <= arch.smem_cap, (
            f"{name} uses {dynamic_smem} bytes of dynamic shared memory, exceeding the device limit of {arch.smem_cap} bytes"
        )
        assert usage is not None, f"{name} is missing HIP compiler resource metadata"
        if usage.n_regs:
            assert usage.n_regs * block_size <= arch.reg_cap, (
                f"{name} uses {usage.n_regs} VGPRs per thread across {block_size} threads "
                f"({usage.n_regs * block_size} total), exceeding the device register-file "
                f"limit of {arch.reg_cap}"
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

    query_ref = query.float().detach().requires_grad_(True)
    key_ref = key.float().detach().requires_grad_(True)
    value_ref = value.float().detach().requires_grad_(True)
    output_ref, _ = linear_attention_bwd.ref_program(query_ref, key_ref, value_ref)
    output_ref.backward(grad_output.float())
    expected = (query_ref.grad, key_ref.grad, value_ref.grad)

    assert len(actual) == len(expected) == 3
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-2, atol=1e-2)
