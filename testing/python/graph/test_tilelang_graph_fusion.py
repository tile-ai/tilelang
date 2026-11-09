"""Fusion must preserve results and require valid, repeatable measurements."""

from types import SimpleNamespace

import pytest
import torch

from tilelang.graph.frontend import fx_to_relax
from tilelang.graph.fusion import (
    GraphValidationError,
    kernel_count,
    make_fusion_candidate,
    output_reduction_flags,
    select_fusion,
    validate_runner,
)
from tilelang.profiler.errors import ProfileUnavailableError


def _options(**kwargs):
    return SimpleNamespace(benchmark_repeats=3, benchmark_backend="event", min_speedup=0.05, max_regression=0.02, **kwargs)


def test_fuse_relax_call_tir_chain():
    class Model(torch.nn.Module):
        def forward(self, x, bias):
            return torch.relu(x + bias) * bias

    inputs = [torch.randn(7, 65), torch.randn(65)]
    mod = fx_to_relax(torch.fx.symbolic_trace(Model()), inputs)
    fused = make_fusion_candidate(mod)
    assert kernel_count(mod) == 3
    assert kernel_count(fused) == 1
    # Grouping must not mutate the independent baseline used for fallback.
    assert kernel_count(mod) == 3


def test_fusion_requires_correctness_and_repeatable_speedup():
    def baseline(x):
        return x + 1

    def candidate(x):
        return x + 1

    def benchmark(runner, inputs, device, backend):
        if backend == "wall":
            return 2.0 if runner is baseline else 1.9
        return 1.0 if runner is baseline else 0.8

    selected = select_fusion(baseline, candidate, [torch.arange(8.0)], _options(), reference=baseline, benchmark=benchmark)
    assert selected.fusion_decision.accepted
    assert selected.runner is candidate
    assert selected.fusion_decision.speedup == pytest.approx(1.25)
    torch.testing.assert_close(selected(torch.ones(3)), torch.full((3,), 2.0))


@pytest.mark.parametrize("sample", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timing_retains_baseline(sample):
    def baseline(x):
        return x + 1

    def candidate(x):
        return x + 1

    selected = select_fusion(baseline, candidate, [torch.ones(8)], _options(), benchmark=lambda *args: sample)
    assert selected.runner is baseline
    assert not selected.fusion_decision.accepted
    assert "timing" in selected.fusion_decision.reason


def test_missing_npu_measurement_retains_baseline():
    def baseline(x):
        return x + 1

    selected = select_fusion(baseline, lambda x: x + 1, [torch.ones(8)], _options())
    assert selected.runner is baseline
    assert "unavailable" in selected.fusion_decision.reason


def test_wrong_candidate_is_rejected_before_benchmark():
    def benchmark(*args):
        pytest.fail("Incorrect fusion candidates must never be benchmarked")

    def baseline(x):
        return x + 1

    selected = select_fusion(baseline, lambda x: x - 1, [torch.ones(8)], _options(), benchmark=benchmark)
    assert selected.runner is baseline
    assert "correctness" in selected.fusion_decision.reason


def test_baseline_is_checked_against_torch_reference():
    with pytest.raises(GraphValidationError, match="correctness"):
        select_fusion(lambda x: x - 1, lambda x: x - 1, [torch.ones(8)], _options(), reference=lambda x: x + 1)


@pytest.mark.parametrize("failure", ["wall_regression", "noisy", "inconsistent", "small_gain"])
def test_fusion_rejects_unreliable_or_regressing_measurements(failure):
    def baseline(x):
        return x + 1

    def candidate(x):
        return x + 1

    calls = 0

    def benchmark(runner, inputs, device, backend):
        nonlocal calls
        calls += 1
        if backend == "wall":
            return 1.1 if failure == "wall_regression" and runner is candidate else 1.0
        if runner is baseline:
            return 1.0
        if failure == "noisy":
            return 0.7 if calls < 8 else 0.8
        if failure == "inconsistent":
            return 0.8 if calls < 8 else 1.1
        return 0.98 if failure == "small_gain" else 0.8

    selected = select_fusion(baseline, candidate, [torch.ones(8)], _options(), benchmark=benchmark)
    assert not selected.fusion_decision.accepted
    assert selected.runner is baseline


def test_device_failures_are_not_converted_into_eager_fallback():
    def broken(x):
        raise RuntimeError("device launch failed")

    with pytest.raises(RuntimeError, match="device launch failed"):
        validate_runner(broken, lambda x: x + 1, [torch.ones(8)])


def test_missing_profile_retains_baseline_but_measurement_device_errors_escape():
    def baseline(x):
        return x + 1

    def unavailable(*args):
        raise ProfileUnavailableError("no target kernels")

    selected = select_fusion(baseline, baseline, [torch.ones(8)], _options(), benchmark=unavailable)
    assert selected.runner is baseline
    assert "unavailable" in selected.fusion_decision.reason

    def device_failure(*args):
        raise RuntimeError("device synchronization failed")

    with pytest.raises(RuntimeError, match="device synchronization failed"):
        select_fusion(baseline, baseline, [torch.ones(8)], _options(), benchmark=device_failure)


@pytest.mark.parametrize("candidate", [lambda x: [x], lambda x: x.half(), lambda x: x[:1]])
def test_validation_checks_structure_dtype_and_shape(candidate):
    with pytest.raises(GraphValidationError):
        validate_runner(candidate, lambda x: x, [torch.ones(8)])


def test_validation_preserves_signed_zero():
    with pytest.raises(GraphValidationError, match="sign of zero"):
        validate_runner(lambda x: -x, lambda x: x, [torch.zeros(8)])


def test_reduction_tolerance_does_not_leak_to_an_independent_output():
    def model(x):
        return x.sum(-1), x + 1

    mod = fx_to_relax(torch.fx.symbolic_trace(model), [torch.zeros(7, 65, dtype=torch.float16)])
    assert output_reduction_flags(mod) == (True, False)

    def runner(x):
        return x.sum(-1), x + 0.005

    runner.has_reduction = True
    runner.output_reductions = (True, False)
    with pytest.raises(GraphValidationError, match="output 1"):
        validate_runner(runner, lambda x: (x.sum(-1), x), [torch.zeros(7, 65, dtype=torch.float16)])
