# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""Routing contracts and four-output device regressions for shifted mHC Pre."""

import math

import pytest
import torch

from examples.ascend.mhc.mhc_pre import _select_kernel, bind_mhc_pre, mhc_pre


@pytest.mark.parametrize(
    "n,hidden,route",
    [
        (2304, 4096, "mk"),
        (2305, 4096, "large"),
        (2400, 4096, "mk"),
        (2401, 4096, "large"),
        (2560, 5120, "mk"),
        (2561, 5120, "large"),
        (3136, 7168, "large"),
        (3584, 7168, "mk"),
        (3585, 7168, "large"),
        (8192, 4096, "large"),
    ],
)
def test_routing_preserves_original_mk_complete_tiles(n, hidden, route):
    assert _select_kernel(n, hidden)["route"] == route


@pytest.mark.parametrize(
    "n,aic,aiv,route",
    [
        (2305, 24, 48, "mk"),
        (3073, 24, 48, "large"),
        (2305, 32, 56, "mk"),
        (3585, 32, 56, "large"),
        (3585, 32, 64, "mk"),
        (4097, 32, 64, "large"),
    ],
)
def test_other_core_capacities_retain_their_partition(n, aic, aiv, route):
    assert _select_kernel(n, 4096, num_aic_cores=aic, num_vec_cores=aiv)["route"] == route


@pytest.mark.parametrize(
    "n,hidden,aic,aiv",
    [
        (0, 4096, 28, 56),
        (-1, 4096, 28, 56),
        (True, 4096, 28, 56),
        (16, 4097, 28, 56),
        (1, 4096, 0, 56),
        (1, 4096, 28, 1),
    ],
)
def test_invalid_geometry_is_rejected(n, hidden, aic, aiv):
    with pytest.raises(ValueError):
        _select_kernel(n, hidden, num_aic_cores=aic, num_vec_cores=aiv)


def _inputs(n, hidden, device, *, zero=False, seed_offset=0):
    generator = torch.Generator(device="cpu").manual_seed(20261112 + n + hidden + seed_offset)
    x = torch.randn((n, 4, hidden), dtype=torch.bfloat16, generator=generator)
    if zero:
        x.zero_()
    previous = torch.randn((n, 4, 1), dtype=torch.float32, generator=generator).sigmoid().add_(1e-6)
    phi = torch.randn((24, 4 * hidden), dtype=torch.float32, generator=generator).div_(math.sqrt(4 * hidden))
    alpha = torch.randn((3,), dtype=torch.float32, generator=generator).mul_(0.1).add_(1)
    bias = torch.randn((24,), dtype=torch.float32, generator=generator).mul_(0.1)
    return tuple(tensor.to(device=device) for tensor in (x, previous, phi, alpha, bias))


def _reference(x, previous, phi, alpha, bias):
    """Independent CPU FP32 equations, with bounded residual intermediates."""
    n, _, hidden = x.shape
    phi, alpha, bias = (tensor.detach().float().cpu() for tensor in (phi, alpha, bias))
    previous = previous.detach().float().cpu().reshape(n, 4, 1)
    layer = torch.empty((n, hidden), dtype=torch.bfloat16)
    pre = torch.empty((n, 4, 1), dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((n, 4, 4), dtype=torch.float32)
    for start in range(0, n, 128):
        end = min(start + 128, n)
        residual = x[start:end].detach().float().cpu()
        flat = residual.reshape(end - start, 4 * hidden)
        mix = (flat @ phi.T) * (flat.square().mean(-1, keepdim=True) + 1e-20).rsqrt()
        pre[start:end, :, 0] = (mix[:, :4] * alpha[0] + bias[:4]).sigmoid() + 1e-6
        post[start:end, :, 0] = 2 * (mix[:, 4:8] * alpha[1] + bias[4:8]).sigmoid()
        matrix = (mix[:, 8:24] * alpha[2] + bias[8:24]).reshape(-1, 4, 4).softmax(-1) + 1e-6
        matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            matrix = matrix / (matrix.sum(-1, keepdim=True) + 1e-6)
            matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
        comb[start:end] = matrix
        layer[start:end] = (residual * previous[start:end]).sum(1).to(torch.bfloat16)
    return layer, pre, post, comb


def _assert_outputs(actual, expected):
    assert isinstance(actual, (tuple, list)) and len(actual) == 4
    for index, (value, reference) in enumerate(zip(actual, expected)):
        assert value.shape == reference.shape
        assert value.dtype == reference.dtype
        assert value.device.type == "npu"
        actual_cpu = value.cpu()
        assert torch.isfinite(actual_cpu).all().item()
        torch.testing.assert_close(actual_cpu, reference, rtol=1e-3, atol=0.03125 if index == 0 else 5e-5)


def test_cpu_inputs_are_rejected_before_device_compilation():
    with pytest.raises(ValueError, match="NPU|npu"):
        bind_mhc_pre(*_inputs(1, 4096, "cpu"))


@pytest.fixture
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize(
    "hidden,n,zero",
    [
        pytest.param(4096, 1, True, id="mk-zero-rms"),
        pytest.param(5120, 33, False, id="mk-row-tail"),
        pytest.param(7168, 129, False, id="mk-coefficient-batches"),
        pytest.param(4096, 2305, False, id="large-empty-aiv-half"),
        pytest.param(5120, 2400, False, id="mk-m240-complete"),
        pytest.param(7168, 2401, False, id="large-after-mk-complete"),
        pytest.param(4096, 3137, False, id="large-aiv-boundary-tail"),
        pytest.param(5120, 3584, False, id="mk-m256-complete"),
        pytest.param(7168, 3585, False, id="large-natural-boundary"),
        pytest.param(4096, 8192, False, id="large-multiple-jobs"),
    ],
)
def test_four_outputs_against_cpu_reference(npu_device, hidden, n, zero):
    inputs = list(_inputs(n, hidden, npu_device, zero=zero))
    if n == 3137:
        inputs[1] = inputs[1].reshape(n, 4)
    expected = _reference(*inputs)
    with torch.inference_mode():
        actual = mhc_pre(*inputs)
    torch.npu.synchronize(npu_device)
    _assert_outputs(actual, expected)


@pytest.mark.parametrize("n,hidden", [(33, 5120), (2305, 4096)], ids=["mk", "large"])
def test_bound_launch_uses_current_stream_and_fresh_outputs(npu_device, n, hidden):
    inputs = _inputs(n, hidden, npu_device)
    run = bind_mhc_pre(*inputs)
    with torch.inference_mode():
        first = run()
        torch.npu.synchronize(npu_device)
        _assert_outputs(first, _reference(*inputs))
        saved_first = tuple(tensor.cpu().clone() for tensor in first)

        stream = torch.npu.Stream(device=npu_device)
        with torch.npu.stream(stream):
            inputs[1].add_(0.125)
            inputs[2].mul_(-0.75)
            inputs[4].add_(0.2)
            second = run()
        stream.synchronize()
        _assert_outputs(second, _reference(*inputs))

    assert all(before.data_ptr() != after.data_ptr() for before, after in zip(first, second))
    for before, saved in zip(first, saved_first):
        torch.testing.assert_close(before.cpu(), saved, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["dtype", "noncontiguous"])
def test_invalid_npu_inputs_are_rejected(npu_device, invalid):
    inputs = list(_inputs(1, 4096, npu_device))
    if invalid == "dtype":
        inputs[0] = inputs[0].float()
    else:
        inputs[2] = torch.empty((24, 8 * 4096), dtype=torch.float32, device=npu_device)[:, ::2]
    with pytest.raises(ValueError):
        bind_mhc_pre(*inputs)


def test_functional_cache_accepts_new_input_objects(npu_device):
    first_inputs = _inputs(33, 5120, npu_device)
    second_inputs = _inputs(33, 5120, npu_device, seed_offset=1)
    with torch.inference_mode():
        first = mhc_pre(*first_inputs)
        second = mhc_pre(*second_inputs)
    torch.npu.synchronize(npu_device)
    _assert_outputs(first, _reference(*first_inputs))
    _assert_outputs(second, _reference(*second_inputs))
    assert all(before.data_ptr() != after.data_ptr() for before, after in zip(first, second))
