# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""Post mixing direction, persistent work assignment and Pre/Post integration."""

import pytest
import torch

from examples.ascend.mhc.mhc_post import mhc_post
from examples.ascend.mhc.mhc_pre import mhc_pre


def _inputs(batch, hidden, device):
    generator = torch.Generator(device="cpu").manual_seed(20261112 + hidden)
    x = torch.randn((*batch, hidden), dtype=torch.bfloat16, generator=generator)
    residual = torch.randn((*batch, 4, hidden), dtype=torch.bfloat16, generator=generator)
    post = torch.randn((*batch, 4, 1), dtype=torch.float32, generator=generator)
    # Deliberately nonsymmetric: transposing the combination direction must fail.
    comb = torch.randn((*batch, 4, 4), dtype=torch.float32, generator=generator)
    return tuple(value.to(device) for value in (x, residual, post, comb))


def _reference(x, residual, post, comb):
    x, residual, post, comb = (value.detach().float().cpu() for value in (x, residual, post, comb))
    post = post.reshape(*residual.shape[:-1], 1)
    return (comb.transpose(-1, -2) @ residual + post * x.unsqueeze(-2)).bfloat16()


def _assert_output(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "npu"
    actual = actual.cpu()
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=1e-3)


@pytest.fixture
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    return torch.device("npu", torch.npu.current_device())


def test_cpu_inputs_are_rejected_before_device_compilation():
    with pytest.raises(ValueError, match="NPU|npu"):
        mhc_post(*_inputs((1,), 4096, "cpu"))


@pytest.mark.parametrize("invalid", ["empty", "hidden", "mix_shape", "cores_zero", "cores_bool"])
def test_invalid_geometry_is_rejected(invalid):
    inputs = list(_inputs((0,) if invalid == "empty" else (1,), 128 if invalid == "hidden" else 4096, "cpu"))
    if invalid == "mix_shape":
        inputs[2] = torch.empty((1, 5), dtype=torch.float32)
    cores = 0 if invalid == "cores_zero" else True if invalid == "cores_bool" else 56
    with pytest.raises(ValueError, match="token count|post_layer_mix|num_vec_cores"):
        mhc_post(*inputs, num_vec_cores=cores)


@pytest.mark.parametrize(
    "batch,hidden,squeeze_mix,cores",
    [
        pytest.param((1,), 4096, False, 56, id="single-token"),
        pytest.param((57,), 4096, True, 56, id="persistent-token-tail"),
        pytest.param((85,), 5120, False, 56, id="two-hidden-tiles-three-stages"),
        pytest.param((2, 17), 7168, False, 56, id="batched-hidden-tiles"),
        pytest.param((17,), 5120, True, 4, id="smaller-core-capacity"),
    ],
)
def test_post_against_cpu_reference(npu_device, batch, hidden, squeeze_mix, cores):
    inputs = list(_inputs(batch, hidden, npu_device))
    if squeeze_mix:
        inputs[2] = inputs[2].squeeze(-1)
    expected = _reference(*inputs)
    with torch.inference_mode():
        actual = mhc_post(*inputs, num_vec_cores=cores)
    torch.npu.synchronize(npu_device)
    _assert_output(actual, expected)


def test_current_stream_fresh_outputs_and_new_inputs(npu_device):
    inputs = _inputs((29,), 5120, npu_device)
    with torch.inference_mode():
        first = mhc_post(*inputs)
        torch.npu.synchronize(npu_device)
        _assert_output(first, _reference(*inputs))
        saved = first.cpu().clone()
        stream = torch.npu.Stream(device=npu_device)
        with torch.npu.stream(stream):
            # New tensor objects at the same shape must reach the cached kernel.
            second_inputs = (inputs[0].neg(), inputs[1], inputs[2] * 0.75, inputs[3])
            second = mhc_post(*second_inputs)
        stream.synchronize()
        _assert_output(second, _reference(*second_inputs))
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), saved, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["dtype", "noncontiguous", "batch_shape", "device"])
def test_invalid_device_inputs_are_rejected(npu_device, invalid):
    inputs = list(_inputs((2,), 4096, npu_device))
    if invalid == "dtype":
        inputs[0] = inputs[0].float()
    elif invalid == "noncontiguous":
        inputs[3] = inputs[3].transpose(-1, -2)
    elif invalid == "batch_shape":
        inputs[0] = inputs[0].view(1, 2, 4096)
    else:
        inputs[3] = inputs[3].cpu()
    with pytest.raises(ValueError):
        mhc_post(*inputs)


def test_pre_outputs_feed_post_on_the_same_stream(npu_device):
    n, hidden = 33, 4096
    x, residual, previous, _ = _inputs((n,), hidden, npu_device)
    phi = torch.randn((24, 4 * hidden), device=npu_device, dtype=torch.float32) / (4 * hidden) ** 0.5
    alpha = torch.ones(3, device=npu_device, dtype=torch.float32)
    bias = torch.zeros(24, device=npu_device, dtype=torch.float32)
    with torch.inference_mode():
        layer_input, _, post, comb = mhc_pre(residual, previous, phi, alpha, bias)
        # A small synthetic layer stands in for the model's attention/MLP.
        layer_output = (layer_input + x).contiguous()
        actual = mhc_post(layer_output, residual, post, comb)
    torch.npu.synchronize(npu_device)
    _assert_output(actual, _reference(layer_output, residual, post, comb))
