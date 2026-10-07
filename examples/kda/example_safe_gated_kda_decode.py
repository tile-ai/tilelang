"""Packed safe-gated KDA decode implemented in TileLang.

This kernel consumes the post-convolution packed QKV projection used by KDA
decode.  It applies Q/K normalization, the bounded ("safe") gate, and the
delta-rule recurrence while updating a slot-indexed state pool in place.

The state layout is ``[num_slots, value_heads, value_dim, key_dim]``.  A state
index of ``-1`` is a dummy request: its output is zero and the state pool is not
accessed or modified.
"""

import torch

import tilelang
import tilelang.language as T


def validate_safe_gated_kda_state_indices(
    state_indices: torch.Tensor,
    *,
    batch: int,
    num_slots: int,
) -> None:
    """Validate state-slot ownership before launching a decode kernel."""
    if state_indices.shape != (batch,):
        raise ValueError("state_indices must have shape [batch]")

    active_indices = state_indices[state_indices >= 0].to(torch.int64)
    if active_indices.numel() and int(active_indices.max()) >= num_slots:
        raise ValueError("state index is outside the state pool")
    if active_indices.numel() and torch.unique(active_indices).numel() != active_indices.numel():
        raise ValueError("active state indices must be unique within a batch")


def safe_gated_kda_decode_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    num_q_heads: int,
    lower_bound: float = -5.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Torch reference for one packed safe-gated KDA decode step.

    ``state`` is updated in place to match the serving-kernel contract.
    Active state indices must be unique within a batch because each request
    owns one mutable recurrent-state slot.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [batch, packed_qkv_dim]")
    if state.ndim != 4:
        raise ValueError("state must have shape [num_slots, value_heads, value_dim, key_dim]")
    if lower_bound >= 0:
        raise ValueError("safe-gate lower_bound must be negative")

    batch = mixed_qkv.shape[0]
    num_value_heads, value_dim, key_dim = state.shape[1:]
    if num_value_heads % num_q_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_q_heads")
    if mixed_qkv.shape[1] != 2 * num_q_heads * key_dim + num_value_heads * value_dim:
        raise ValueError("mixed_qkv has an incompatible packed dimension")
    if a.shape != (batch, num_value_heads * key_dim):
        raise ValueError("a must have shape [batch, value_heads * key_dim]")
    if b.shape != (batch, num_value_heads):
        raise ValueError("b must have shape [batch, value_heads]")
    if A_log.shape != (num_value_heads,):
        raise ValueError("A_log must have shape [value_heads]")
    if dt_bias.shape != (num_value_heads * key_dim,):
        raise ValueError("dt_bias must have shape [value_heads * key_dim]")
    validate_safe_gated_kda_state_indices(
        state_indices,
        batch=batch,
        num_slots=state.shape[0],
    )

    if scale is None:
        scale = key_dim**-0.5

    output = mixed_qkv.new_zeros((batch, 1, num_value_heads, value_dim))
    q_end = num_q_heads * key_dim
    k_end = 2 * q_end
    q = mixed_qkv[:, :q_end].view(batch, num_q_heads, key_dim).float()
    k = mixed_qkv[:, q_end:k_end].view(batch, num_q_heads, key_dim).float()
    v = mixed_qkv[:, k_end:].view(batch, num_value_heads, value_dim).float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * scale

    head_group_size = num_value_heads // num_q_heads
    for batch_idx in range(batch):
        slot = int(state_indices[batch_idx])
        if slot < 0:
            continue
        for value_head in range(num_value_heads):
            qk_head = value_head // head_group_size
            gate_start = value_head * key_dim
            gate_end = gate_start + key_dim
            gate_input = a[batch_idx, gate_start:gate_end].float() + dt_bias[gate_start:gate_end].float()
            log_decay = lower_bound * torch.sigmoid(torch.exp(A_log[value_head].float()) * gate_input)
            beta = torch.sigmoid(b[batch_idx, value_head].float())

            state_head = state[slot, value_head].float()
            state_head = state_head * torch.exp(log_decay).unsqueeze(0)
            delta = (v[batch_idx, value_head] - (state_head * k[batch_idx, qk_head]).sum(dim=-1)) * beta
            state_head = state_head + delta.unsqueeze(-1) * k[batch_idx, qk_head].unsqueeze(0)
            output[batch_idx, 0, value_head] = (state_head * q[batch_idx, qk_head]).sum(dim=-1).to(output.dtype)
            state[slot, value_head] = state_head.to(state.dtype)

    return output


@tilelang.jit(out_idx=[7])
def safe_gated_kda_decode(
    batch: int,
    num_slots: int,
    num_q_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    input_dtype: str = "bfloat16",
    state_dtype: str = "float32",
    lower_bound: float = -5.0,
    scale: float | None = None,
    block_v: int = 16,
    threads: int = 128,
):
    """Create a packed safe-gated KDA decode kernel.

    GLM-5.3-Flash TP8 uses ``num_q_heads=num_value_heads=8`` and
    ``key_dim=value_dim=128``.  The state defaults to FP32, matching SGLang's
    production temporal-state default.  ``block_v`` must divide ``value_dim``.
    """
    if batch <= 0 or num_slots <= 0:
        raise ValueError("batch and num_slots must be positive")
    if num_q_heads <= 0 or num_value_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_value_heads % num_q_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_q_heads")
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("key_dim and value_dim must be positive")
    if block_v <= 0 or value_dim % block_v != 0:
        raise ValueError("block_v must divide value_dim")
    if lower_bound >= 0:
        raise ValueError("safe-gate lower_bound must be negative")
    if scale is None:
        scale = key_dim**-0.5

    packed_qkv_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    head_group_size = num_value_heads // num_q_heads
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        MixedQKV: T.Tensor((batch, packed_qkv_dim), input_dtype),
        A: T.Tensor((batch, num_value_heads * key_dim), input_dtype),
        B: T.Tensor((batch, num_value_heads), input_dtype),
        A_log: T.Tensor((num_value_heads,), T.float32),
        DTBias: T.Tensor((num_value_heads * key_dim,), T.float32),
        State: T.Tensor((num_slots, num_value_heads, value_dim, key_dim), state_dtype),
        StateIndices: T.Tensor((batch,), T.int32),
        Output: T.Tensor((batch, 1, num_value_heads, value_dim), input_dtype),
    ):
        with T.Kernel(value_dim // block_v, batch, num_value_heads, threads=threads) as (
            i_v,
            i_b,
            i_hv,
        ):
            state_slot = T.alloc_var(T.int64)
            state_slot = T.Cast(T.int64, StateIndices[i_b])
            value_offset = i_v * block_v

            # The host contract rejects positive indices outside the pool.  Keep
            # the device path defensive as well: an invalid slot must never be
            # used to address State, even if a caller bypasses host validation.
            if state_slot < 0 or state_slot >= num_slots:
                for i in T.Parallel(block_v):
                    Output[i_b, 0, i_hv, value_offset + i] = T.Cast(input_dtype, 0.0)
            else:
                qk_head = i_hv // head_group_size
                q_offset = qk_head * key_dim
                k_offset = num_q_heads * key_dim + q_offset
                v_offset = 2 * num_q_heads * key_dim + i_hv * value_dim + value_offset
                gate_offset = i_hv * key_dim

                q = T.alloc_fragment((1, key_dim), accum_dtype)
                k = T.alloc_fragment((1, key_dim), accum_dtype)
                norm_terms = T.alloc_fragment((1, key_dim), accum_dtype)
                norm = T.alloc_fragment((1,), accum_dtype)
                decay = T.alloc_fragment((1, key_dim), accum_dtype)
                h = T.alloc_fragment((block_v, key_dim), accum_dtype)
                products = T.alloc_fragment((block_v, key_dim), accum_dtype)
                delta = T.alloc_fragment((block_v,), accum_dtype)
                output = T.alloc_fragment((block_v,), accum_dtype)
                beta = T.alloc_var(accum_dtype)
                a_log = T.alloc_var(accum_dtype)

                for j in T.Parallel(key_dim):
                    q[0, j] = T.Cast(accum_dtype, MixedQKV[i_b, q_offset + j])
                    norm_terms[0, j] = q[0, j] * q[0, j]
                T.reduce_sum(norm_terms, norm, dim=1)
                for j in T.Parallel(key_dim):
                    q[0, j] = q[0, j] * T.rsqrt(norm[0] + 1e-6) * scale

                for j in T.Parallel(key_dim):
                    k[0, j] = T.Cast(accum_dtype, MixedQKV[i_b, k_offset + j])
                    norm_terms[0, j] = k[0, j] * k[0, j]
                T.reduce_sum(norm_terms, norm, dim=1)
                for j in T.Parallel(key_dim):
                    k[0, j] = k[0, j] * T.rsqrt(norm[0] + 1e-6)

                a_log = T.Cast(accum_dtype, A_log[i_hv])
                beta = T.sigmoid(T.Cast(accum_dtype, B[i_b, i_hv]))
                for j in T.Parallel(key_dim):
                    gate_input = T.Cast(accum_dtype, A[i_b, gate_offset + j]) + T.Cast(accum_dtype, DTBias[gate_offset + j])
                    log_decay = lower_bound * T.sigmoid(T.exp(a_log) * gate_input)
                    decay[0, j] = T.exp(log_decay)

                for i, j in T.Parallel(block_v, key_dim):
                    h[i, j] = (
                        T.Cast(
                            accum_dtype,
                            State[state_slot, i_hv, value_offset + i, j],
                        )
                        * decay[0, j]
                    )
                    products[i, j] = h[i, j] * k[0, j]
                T.reduce_sum(products, delta, dim=1)

                for i in T.Parallel(block_v):
                    delta[i] = (T.Cast(accum_dtype, MixedQKV[i_b, v_offset + i]) - delta[i]) * beta
                for i, j in T.Parallel(block_v, key_dim):
                    h[i, j] = h[i, j] + delta[i] * k[0, j]
                    products[i, j] = h[i, j] * q[0, j]
                T.reduce_sum(products, output, dim=1)

                for i, j in T.Parallel(block_v, key_dim):
                    State[state_slot, i_hv, value_offset + i, j] = T.Cast(state_dtype, h[i, j])
                for i in T.Parallel(block_v):
                    Output[i_b, 0, i_hv, value_offset + i] = T.Cast(input_dtype, output[i])

    return kernel


def run_safe_gated_kda_decode(
    kernel,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Validate mutable-slot ownership and launch a specialized KDA kernel."""
    validate_safe_gated_kda_state_indices(
        state_indices,
        batch=mixed_qkv.shape[0],
        num_slots=state.shape[0],
    )
    return kernel(mixed_qkv, a, b, A_log, dt_bias, state, state_indices)


def _run_example() -> None:
    """Run one GLM-5.3-shaped correctness check on the active accelerator."""
    torch.manual_seed(0)
    batch, num_slots = 4, 7
    num_q_heads = num_value_heads = 8
    key_dim = value_dim = 128
    device = "cuda"

    mixed_qkv = torch.randn(
        batch,
        2 * num_q_heads * key_dim + num_value_heads * value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    a = torch.randn(batch, num_value_heads * key_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(batch, num_value_heads, device=device, dtype=torch.bfloat16)
    A_log = torch.linspace(-2.0, 0.0, num_value_heads, device=device)
    dt_bias = torch.randn(num_value_heads * key_dim, device=device) * 0.1
    state_indices = torch.tensor([5, -1, 1, 3], device=device, dtype=torch.int32)
    initial_state = torch.randn(
        num_slots,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )

    expected_state = initial_state.clone()
    expected = safe_gated_kda_decode_reference(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        expected_state,
        state_indices,
        num_q_heads=num_q_heads,
    )
    actual_state = initial_state.clone()
    kernel = safe_gated_kda_decode(
        batch,
        num_slots,
        num_q_heads,
        num_value_heads,
        key_dim,
        value_dim,
    )
    actual = run_safe_gated_kda_decode(
        kernel,
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        actual_state,
        state_indices,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-3, atol=2e-3)
    print("Safe-gated KDA decode checks pass.")


if __name__ == "__main__":
    _run_example()
