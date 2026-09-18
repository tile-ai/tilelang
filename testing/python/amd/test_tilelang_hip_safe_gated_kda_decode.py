import sys
from pathlib import Path

import pytest
import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "kda"
sys.path.insert(0, str(_EXAMPLE_DIR))

from example_safe_gated_kda_decode import (  # noqa: E402
    run_safe_gated_kda_decode,
    safe_gated_kda_decode,
    safe_gated_kda_decode_reference,
)


def _make_inputs(
    *,
    batch: int,
    num_slots: int,
    num_q_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    seed: int,
):
    """Create deterministic BF16 inputs and an FP32 recurrent-state pool."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed_dim = 2 * num_q_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = torch.randn(
        batch,
        packed_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    a = torch.randn(
        batch,
        num_value_heads * key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    b = torch.randn(
        batch,
        num_value_heads,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    A_log = torch.linspace(-2.0, 0.0, num_value_heads, device="cuda", dtype=torch.float32)
    dt_bias = (
        torch.randn(
            num_value_heads * key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    )
    state = (
        torch.randn(
            num_slots,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    )
    return mixed_qkv, a, b, A_log, dt_bias, state


def _assert_step(
    kernel,
    inputs,
    state_indices: torch.Tensor,
    actual_state: torch.Tensor,
    expected_state: torch.Tensor,
    *,
    num_q_heads: int,
    lower_bound: float,
):
    """Run one decode step and compare output plus the in-place state update."""
    mixed_qkv, a, b, A_log, dt_bias, _ = inputs
    expected = safe_gated_kda_decode_reference(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        expected_state,
        state_indices,
        num_q_heads=num_q_heads,
        lower_bound=lower_bound,
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
    return actual


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("lower_bound", [-5.0, -3.0])
def test_safe_gated_kda_decode_state_slots_and_multiple_steps(lower_bound):
    """Check active, dummy, untouched, and reused slots over two decode steps."""
    batch, num_slots = 4, 7
    num_q_heads, num_value_heads = 2, 4
    key_dim = value_dim = 32
    inputs = _make_inputs(
        batch=batch,
        num_slots=num_slots,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seed=7,
    )
    state_indices = torch.tensor([5, -1, 1, 3], device="cuda", dtype=torch.int32)
    inputs[1][0, :4] = torch.tensor([-80.0, -20.0, 20.0, 80.0], device="cuda", dtype=torch.bfloat16)
    initial_state = inputs[-1]
    actual_state = initial_state.clone()
    expected_state = initial_state.clone()
    untouched_state = initial_state.clone()

    kernel = safe_gated_kda_decode(
        batch,
        num_slots,
        num_q_heads,
        num_value_heads,
        key_dim,
        value_dim,
        lower_bound=lower_bound,
        block_v=8,
        threads=128,
    )
    actual = _assert_step(
        kernel,
        inputs,
        state_indices,
        actual_state,
        expected_state,
        num_q_heads=num_q_heads,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)
    for slot in (0, 2, 4, 6):
        torch.testing.assert_close(actual_state[slot], untouched_state[slot], rtol=0, atol=0)

    second_inputs = _make_inputs(
        batch=batch,
        num_slots=num_slots,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seed=19,
    )
    second_indices = torch.tensor([3, 0, -1, 6], device="cuda", dtype=torch.int32)
    _assert_step(
        kernel,
        second_inputs,
        second_indices,
        actual_state,
        expected_state,
        num_q_heads=num_q_heads,
        lower_bound=lower_bound,
    )


@tilelang.testing.requires_rocm
def test_safe_gated_kda_decode_glm53_tp8_shape():
    """Check the eight-local-head GLM-5.3-Flash TP8 specialization."""
    batch, num_slots = 2, 4
    num_q_heads = num_value_heads = 8
    key_dim = value_dim = 128
    lower_bound = -5.0
    inputs = _make_inputs(
        batch=batch,
        num_slots=num_slots,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seed=53,
    )
    state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    initial_state = inputs[-1]
    actual_state = initial_state.clone()
    expected_state = initial_state.clone()

    kernel = safe_gated_kda_decode(
        batch,
        num_slots,
        num_q_heads,
        num_value_heads,
        key_dim,
        value_dim,
        lower_bound=lower_bound,
        block_v=16,
        threads=128,
    )
    _assert_step(
        kernel,
        inputs,
        state_indices,
        actual_state,
        expected_state,
        num_q_heads=num_q_heads,
        lower_bound=lower_bound,
    )


@tilelang.testing.requires_rocm
def test_safe_gated_kda_decode_out_of_range_slot_is_guarded():
    """Ensure a positive out-of-range slot cannot access or modify state."""
    batch, num_slots = 1, 2
    num_q_heads = num_value_heads = 2
    key_dim = value_dim = 32
    inputs = _make_inputs(
        batch=batch,
        num_slots=num_slots,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seed=71,
    )
    state_indices = torch.tensor([num_slots], device="cuda", dtype=torch.int32)
    initial_state = inputs[-1]
    actual_state = initial_state.clone()

    with pytest.raises(ValueError, match="outside the state pool"):
        safe_gated_kda_decode_reference(
            *inputs[:-1],
            initial_state.clone(),
            state_indices,
            num_q_heads=num_q_heads,
        )

    kernel = safe_gated_kda_decode(
        batch,
        num_slots,
        num_q_heads,
        num_value_heads,
        key_dim,
        value_dim,
        block_v=8,
        threads=128,
    )
    with pytest.raises(ValueError, match="outside the state pool"):
        run_safe_gated_kda_decode(kernel, *inputs[:-1], actual_state, state_indices)

    # The raw kernel also prevents an invalid slot from reaching State if a
    # trusted caller bypasses the host-side contract check.
    actual = kernel(*inputs[:-1], actual_state, state_indices)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    torch.testing.assert_close(actual_state, initial_state, rtol=0, atol=0)


@tilelang.testing.requires_rocm
def test_safe_gated_kda_decode_duplicate_active_slots_are_rejected():
    """Reject duplicate active slots before launching parallel state updates."""
    batch, num_slots = 2, 2
    num_q_heads = num_value_heads = 2
    key_dim = value_dim = 32
    inputs = _make_inputs(
        batch=batch,
        num_slots=num_slots,
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        seed=89,
    )
    state_indices = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    kernel = safe_gated_kda_decode(
        batch,
        num_slots,
        num_q_heads,
        num_value_heads,
        key_dim,
        value_dim,
        block_v=8,
        threads=128,
    )

    with pytest.raises(ValueError, match="unique within a batch"):
        run_safe_gated_kda_decode(kernel, *inputs[:-1], inputs[-1], state_indices)
