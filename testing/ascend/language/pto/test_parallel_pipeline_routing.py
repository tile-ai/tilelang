"""Routing tests for the PTO Parallel pipeline skeleton.

Scope: this file proves routing only.

* a PTO ``SimdVF + Parallel`` kernel is routed through
  ``VerifyParallelToPTO -> LegalizeParallelToPTO -> VectorizeParallelToPTO``
  in that order, and never through ``AscendSimdVFLowerParallel``;
* the AscendC SIMD_VF route keeps ``AscendSimdVFLowerParallel`` and never sees
  the three PTO Parallel passes;
* PTO ``SimtVF`` and hand-written PTO VMI modules without a ``T.Parallel``
  unit enter the same three PTO pass entry points, but are bypassed inside
  those passes and accepted without conversion;
* ``PlanPtoSlotStorage`` is neither registered nor called.

This file deliberately asserts routing only: no PTO Parallel semantics, no
lowering output, and no Fragment/slot, control-flow, or Reduce support. Use
``TL_PTO_ROUTING_TRACE_DIR=<dir>`` to dump the per-case ordered pass trace and
the lowered kernel source as review artifacts.

The lowering runs through :func:`tilelang.engine.lower.lower`: TileLang IR
lowering only, with no PTOAS/Bisheng device compilation and no device
execution.
"""

from __future__ import annotations

import os

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import (
    normalize_ascend_target,
    normalize_asc_target,
    normalize_pto_target,
)
from tilelang.engine.lower import lower as _lower
from tilelang.instrumentation import (
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    compile_pass_instrumentation,
    create_pass_instruments,
)
import tvm

_PTO_PARALLEL_PASSES = (
    "tl.VerifyParallelToPTO",
    "tl.LegalizeParallelToPTO",
    "tl.VectorizeParallelToPTO",
)
_ASCENDC_SIMDVF_PASS = "tl.AscendSimdVFLowerParallel"
_SLOT_STORAGE_PASS = "PlanPtoSlotStorage"


class _PassRecorder(PassEventObserver):
    """Records the ordered pass names observed in one lowering."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def pass_started(self, mod, event):
        self.names.append(event.name)
        return None


class _PassTraceTool(PassInstrumentationTool):
    """Compile-session tool that installs the routing recorder."""

    def __init__(self, recorder: _PassRecorder) -> None:
        self._recorder = recorder

    def create_pass_instrument(self):
        return StackedPassInstrument(self._recorder)


def _resolve_target(name: str):
    if name == "pto":
        return normalize_pto_target("pto")
    if name == "ascend":
        return normalize_ascend_target("ascend")
    if name == "asc":
        return normalize_asc_target("asc")
    raise ValueError(f"unsupported target name in routing test: {name}")


def _lower_with_trace(func, target_name: str):
    """Lower *func* for *target_name* while recording the executed pass names."""
    recorder = _PassRecorder()
    # The session must be active before the instruments are created, so the two
    # context managers are entered in this order.
    with (
        compile_pass_instrumentation(name="pto-routing-test", tools=[_PassTraceTool(recorder)]),
        tvm.transform.PassContext(instruments=create_pass_instruments(context="pto-routing-test")),
    ):
        target = _resolve_target(target_name)
        with target:
            artifact = _lower(func, target=target)
    return artifact, recorder.names


def _maybe_dump_artifacts(case: str, names: list[str], artifact) -> None:
    """Dump the pass trace and lowered source when the artifact dir is set."""
    out_dir = os.environ.get("TL_PTO_ROUTING_TRACE_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{case}.pass-trace.txt"), "w") as handle:
        handle.write("\n".join(names) + "\n")
    source = getattr(artifact, "kernel_source", None)
    if source:
        with open(os.path.join(out_dir, f"{case}.source.txt"), "w") as handle:
            handle.write(source)


def _assert_ordered(names: list[str], expected: tuple[str, ...]) -> None:
    for pass_name in expected:
        assert pass_name in names, f"{pass_name} was not executed; trace tail: {names[-12:]}"
    positions = [names.index(pass_name) for pass_name in expected]
    assert positions == sorted(positions), f"expected order {' -> '.join(expected)}, got positions {positions}"


def _simdvf_parallel_kernel(n: int = 256):
    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + b_ub[i]
            T.copy(c_ub, C)

    return main


def _handwritten_vmi_kernel(n: int = 64):
    """SimdVF with hand-written VMI calls and no ``T.Parallel`` unit."""

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(n, size=n)
                c = T.vmi.vload(c_ub[0], size=n)
                a = T.vmi.vload(a_ub[0], size=n)
                b = T.vmi.vload(b_ub[0], size=n)
                out = T.vmi.vadd(T.vmi.vmul(c, a, mask), b, mask)
                T.vmi.vstore(out, c_ub[0], mask)
            T.copy(c_ub, C)

    return main


def _simt_kernel(n: int = 2):
    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(n):
                    c_ub[i] = a_ub[i] + b_ub[i]
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_pto_simdvf_parallel_routes_verify_legalize_vectorize_in_order():
    artifact, names = _lower_with_trace(_simdvf_parallel_kernel(), "pto")
    _assert_ordered(names, _PTO_PARALLEL_PASSES)
    assert _ASCENDC_SIMDVF_PASS not in names, "the PTO route must not run the AscendC SIMD_VF lowering"
    _maybe_dump_artifacts("pto-simdvf-parallel", names, artifact)


def test_ascendc_simdvf_keeps_ascend_simdvf_lower_parallel():
    artifact, names = _lower_with_trace(_simdvf_parallel_kernel(), "ascend")
    assert _ASCENDC_SIMDVF_PASS in names, f"the AscendC route must keep AscendSimdVFLowerParallel; trace tail: {names[-12:]}"
    for pass_name in _PTO_PARALLEL_PASSES:
        assert pass_name not in names, f"the AscendC route must not run {pass_name}"
    _maybe_dump_artifacts("ascendc-simdvf-parallel", names, artifact)


@pytest.mark.pto
def test_pto_simt_is_accepted_by_the_pto_route():
    artifact, names = _lower_with_trace(_simt_kernel(), "pto")
    for pass_name in _PTO_PARALLEL_PASSES:
        assert pass_name in names, f"{pass_name} missing on the PTO route; tail: {names[-12:]}"
    _maybe_dump_artifacts("pto-simt", names, artifact)


@pytest.mark.pto
def test_pto_handwritten_vmi_without_parallel_is_accepted():
    artifact, names = _lower_with_trace(_handwritten_vmi_kernel(), "pto")
    for pass_name in _PTO_PARALLEL_PASSES:
        assert pass_name in names, f"{pass_name} missing on the PTO route; tail: {names[-12:]}"
    _maybe_dump_artifacts("pto-handwritten-vmi", names, artifact)


@pytest.mark.pto
def test_plan_pto_slot_storage_is_not_registered_or_called():
    assert not hasattr(ascend_transform, _SLOT_STORAGE_PASS), (
        "PlanPtoSlotStorage belongs to a later pipeline stage and must not be registered yet"
    )
    artifact, names = _lower_with_trace(_simdvf_parallel_kernel(), "pto")
    assert not any(_SLOT_STORAGE_PASS in name for name in names), "no PlanPtoSlotStorage step may run"
    _maybe_dump_artifacts("pto-slot-storage-absence", names, artifact)
