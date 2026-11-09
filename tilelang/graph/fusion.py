"""Conservative TIR fusion and measured selection for Torch graph execution.

Relax owns grouping and buffer substitution.  The Ascend scheduler must still
prove that each resulting PrimFunc has a supported local execution schedule;
FuseTIR alone does not establish on-chip placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils import _pytree

from tilelang import tvm
from tilelang.profiler.errors import ProfileUnavailableError
from tvm import relax, tirx


class GraphValidationError(RuntimeError):
    """A compiled graph disagrees with its Torch reference."""


def make_fusion_candidate(mod: tvm.IRModule) -> tvm.IRModule:
    """Group pure call_tir producers before target scheduling.

    This deliberately uses the TVM revision shipped with TileLang instead of
    importing the reference PR's copied FuseTIR or scheduled-block canonicalizer.
    VF regions do not exist yet, so this pass cannot flatten their boundaries.
    Unsupported fused groups are rejected by target scheduling and the caller
    retains its independently compiled baseline.
    """
    return tvm.transform.Sequential(
        [
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            relax.transform.DeadCodeElimination(),
        ]
    )(mod)


def kernel_count(mod: tvm.IRModule) -> int:
    """Count actual call sites, rather than unused PrimFunc definitions."""
    count = 0

    def visit(expr):
        nonlocal count
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op) and expr.op.name == "relax.call_tir":
            count += 1

    relax.analysis.post_order_visit(mod["main"].body, visit)
    return count


def has_reduction(mod: tvm.IRModule) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.SBlock) and any(iv.iter_type == tirx.IterVar.CommReduce for iv in node.iter_vars):
            found = True

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def output_reduction_flags(mod: tvm.IRModule) -> tuple[bool, ...]:
    """Track reduction dependence per output, not per graph.

    An unrelated injective output must not inherit a reduction's looser
    tolerance merely because both appear in the same graph result tuple.
    """
    reductions = {gv: has_reduction(tvm.IRModule({gv: func})) for gv, func in mod.functions.items() if isinstance(func, tirx.PrimFunc)}
    env = {param: False for param in mod["main"].params}

    def evaluate(expr):
        if isinstance(expr, relax.Var):
            return env[expr]
        if isinstance(expr, relax.Tuple):
            return tuple(evaluate(field) for field in expr.fields)
        if isinstance(expr, relax.TupleGetItem):
            return evaluate(expr.tuple_value)[expr.index]
        if isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    env[binding.var] = evaluate(binding.value)
            return evaluate(expr.body)
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op) and expr.op.name == "relax.call_tir":
            dependencies, _ = _pytree.tree_flatten(evaluate(expr.args[1]))
            reduced = reductions.get(expr.args[0], False) or any(dependencies)
            info = expr.struct_info
            return tuple(reduced for _ in info.fields) if isinstance(info, relax.TupleStructInfo) else reduced
        return False

    flags, _ = _pytree.tree_flatten(evaluate(mod["main"].body))
    return tuple(flags)


def _snapshot(value):
    return _pytree.tree_map(lambda leaf: leaf.detach().clone() if isinstance(leaf, torch.Tensor) else leaf, value)


def _assert_outputs_close(actual, expected, *, reduction: bool = False, output_reductions=None):
    actual_leaves, actual_spec = _pytree.tree_flatten(actual)
    expected_leaves, expected_spec = _pytree.tree_flatten(expected)
    if actual_spec != expected_spec:
        raise GraphValidationError(f"Compiled graph changed its output structure: {actual_spec} != {expected_spec}")
    if output_reductions is not None and len(output_reductions) != len(actual_leaves):
        raise GraphValidationError("Compiled graph has inconsistent output validation metadata")
    for index, (value, reference) in enumerate(zip(actual_leaves, expected_leaves)):
        if isinstance(reference, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                raise GraphValidationError(f"Output {index} is not a Tensor")
            reduced = bool(output_reductions[index]) if output_reductions is not None else reduction
            tolerances = {
                torch.float32: (1e-4, 1e-4 if reduced else 1e-5),
                torch.float16: (1e-2, 1e-2) if reduced else (1e-3, 1e-3),
                torch.bfloat16: (2e-2, 2e-2) if reduced else (1e-2, 1e-2),
            }
            rtol, atol = tolerances.get(reference.dtype, (0, 0))
            try:
                torch.testing.assert_close(value, reference, rtol=rtol, atol=atol, equal_nan=True)
            except AssertionError as error:
                raise GraphValidationError(f"Compiled graph output {index} failed correctness validation: {error}") from error
            if reference.is_floating_point():
                zero = (value == 0) & (reference == 0)
                if not torch.equal(torch.signbit(value)[zero], torch.signbit(reference)[zero]):
                    raise GraphValidationError(f"Compiled graph output {index} changed the sign of zero")
        elif type(value) is not type(reference) or value != reference:
            raise GraphValidationError(f"Compiled graph changed scalar output {index}: {value!r} != {reference!r}")


def validate_runner(runner: Callable, reference: Callable, inputs: Sequence[Any], *, config=None) -> None:
    """Check a pure inference runner on real inputs, before accepting it.

    Runtime/device errors intentionally escape. Only numerical or output-contract
    mismatches become GraphValidationError, which permits a safe eager fallback.
    """
    with torch.no_grad():
        expected = _snapshot(reference(*inputs))
        actual = runner(*inputs)
        _assert_outputs_close(
            actual,
            expected,
            reduction=bool(getattr(runner, "has_reduction", False)),
            output_reductions=getattr(runner, "output_reductions", None),
        )


@dataclass(frozen=True)
class FusionDecision:
    accepted: bool
    reason: str
    baseline_device_ms: tuple[float, ...] = ()
    candidate_device_ms: tuple[float, ...] = ()
    baseline_wall_ms: tuple[float, ...] = ()
    candidate_wall_ms: tuple[float, ...] = ()

    @property
    def speedup(self) -> float | None:
        if not self.baseline_device_ms or not self.candidate_device_ms:
            return None
        return statistics.median(self.baseline_device_ms) / statistics.median(self.candidate_device_ms)


class SelectedRunner:
    """A selected callable with inspectable correctness/performance evidence."""

    def __init__(self, runner: Callable, decision: FusionDecision):
        self.runner = runner
        self.fusion_decision = decision

    def __call__(self, *args):
        return self.runner(*args)

    def __getattr__(self, name):
        return getattr(self.runner, name)


def _measure(runner: Callable, inputs, device, backend: str) -> float:
    from tilelang.profiler import do_bench

    return do_bench(
        lambda: runner(*inputs),
        device=device,
        backend=backend,
        warmup=10,
        rep=30,
        early_stop_baseline=None,
        return_mode="median",
    )


def select_fusion(
    baseline: Callable,
    candidate: Callable,
    example_inputs: Sequence[Any],
    config,
    *,
    reference: Callable | None = None,
    benchmark: Callable | None = None,
) -> SelectedRunner:
    """Require numerical validity, repeatable speedup, and graph non-regression.

    Alternate execution order to reduce drift. Empty/non-finite measurements
    never imply a win. A caller supplying a benchmark must propagate device
    failures; returning None represents unavailable measurement.
    """
    reference = baseline if reference is None else reference
    validate_runner(baseline, reference, example_inputs, config=config)
    try:
        validate_runner(candidate, reference, example_inputs, config=config)
    except GraphValidationError as error:
        return SelectedRunner(baseline, FusionDecision(False, str(error)))

    tensor_inputs = [value for value in example_inputs if isinstance(value, torch.Tensor)]
    if not tensor_inputs:
        return SelectedRunner(baseline, FusionDecision(False, "no tensor device for measurement"))
    device = tensor_inputs[0].device
    if benchmark is None and device.type != "npu":
        return SelectedRunner(baseline, FusionDecision(False, "NPU measurement unavailable"))
    benchmark = _measure if benchmark is None else benchmark
    repeats = getattr(config, "benchmark_repeats", 3)
    if repeats < 3:
        raise ValueError("Fusion selection requires at least three complete measurement rounds")
    backend = getattr(config, "benchmark_backend", "event")
    if backend not in ("event", "msprof"):
        raise ValueError("Fusion device timing requires event or msprof; wall time is measured separately")
    values = {"baseline_device_ms": [], "candidate_device_ms": [], "baseline_wall_ms": [], "candidate_wall_ms": []}
    with torch.no_grad():
        for round_index in range(repeats):
            order = (("baseline", baseline), ("candidate", candidate))
            if round_index % 2:
                order = tuple(reversed(order))
            for name, runner in order:
                for boundary, method in (("device", backend), ("wall", "wall")):
                    try:
                        value = benchmark(runner, example_inputs, device, method)
                    except ProfileUnavailableError as error:
                        return SelectedRunner(baseline, FusionDecision(False, f"measurement unavailable: {error}"))
                    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        return SelectedRunner(baseline, FusionDecision(False, "invalid or missing timing sample"))
                    values[f"{name}_{boundary}_ms"].append(float(value))
    evidence = {key: tuple(samples) for key, samples in values.items()}
    baseline_device = values["baseline_device_ms"]
    candidate_device = values["candidate_device_ms"]
    baseline_wall = values["baseline_wall_ms"]
    candidate_wall = values["candidate_wall_ms"]
    # A noisy median must not override inconsistent paired results.
    if any(candidate >= base for base, candidate in zip(baseline_device, candidate_device)):
        return SelectedRunner(baseline, FusionDecision(False, "device speedup is not repeatable", **evidence))
    noise_limit = getattr(config, "max_regression", 0.02)
    for samples in values.values():
        if (max(samples) - min(samples)) / statistics.median(samples) > noise_limit:
            return SelectedRunner(baseline, FusionDecision(False, "timing variability exceeds the acceptance boundary", **evidence))
    improvement = 1 - statistics.median(candidate_device) / statistics.median(baseline_device)
    if improvement < getattr(config, "min_speedup", 0.05):
        return SelectedRunner(baseline, FusionDecision(False, "device improvement is below the configured threshold", **evidence))
    if statistics.median(candidate_wall) > statistics.median(baseline_wall) * (1 + noise_limit):
        return SelectedRunner(baseline, FusionDecision(False, "end-to-end graph regression", **evidence))
    return SelectedRunner(candidate, FusionDecision(True, "correct with repeatable device and graph timing", **evidence))
