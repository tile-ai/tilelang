"""Loop-shell tests: constant-bound T.serial inside T.Parallel (Form B).

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``VectorizeParallelToPTO`` (no PTOAS/device compile unless the case is
explicitly numeric).

Expected contract after this stage:

* a constant-bound or runtime-bound ``T.serial`` loop inside ``T.Parallel`` is
  admitted by Verify as a preserved shell and copied through as a scalar loop;
  only its body is vectorized. The tests below pin the shell's presence and
  that a runtime extent stays a runtime expression -- min/step/annotations/
  span fidelity is not asserted here;
* the shell keeps its UB round-trip: a value read and written inside the loop
  stays in its shared buffer and every iteration loads and stores it. The
  per-lane register promotion an earlier draft assumed is not implemented -- the
  current PTOAS inserts no automatic mem_bar, so that form is not order-safe;
* a shell without a memory-carried value (no buffer both read and written
  inside the loop) is preserved as-is;
* a shell whose bounds reference the lane variable, a variable bound inside the
  unit, a buffer load, or a variable that is never defined is rejected by
  Verify with a ``[VerifyParallelToPTO] loop structure`` diagnostic.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm


@contextmanager
def _stage_capture():
    captured = {}
    original_legalize = ascend_transform.LegalizeParallelToPTO
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def capturing_legalize_factory():
        legalize_pass = original_legalize()

        def run(mod):
            result = legalize_pass(mod)
            captured["legalize"] = result
            return result

        return run

    def capturing_vectorize_factory():
        vectorize_pass = original_vectorize()

        def run(mod):
            result = vectorize_pass(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.LegalizeParallelToPTO = capturing_legalize_factory
    ascend_transform.VectorizeParallelToPTO = capturing_vectorize_factory
    try:
        yield captured
    finally:
        ascend_transform.LegalizeParallelToPTO = original_legalize
        ascend_transform.VectorizeParallelToPTO = original_vectorize


def _lower_capture(func, *, expect_error=False):
    target = normalize_pto_target("pto")
    error = None
    with _stage_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured, error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _serial_shells_named(mod, name):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and node.loop_var.name == name:
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_newton_inside(E, lanes, dtype="float32"):
    """Seed store + 3-step Newton self-RMW on y_ub: the canonical carry."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(1.0)
                    for _t in T.serial(3):
                        y_ub[i] = y_ub[i] * (T.float32(2.0) - x_ub[i] * y_ub[i])
            T.copy(y_ub, Y)

    return main


def _k_runtime_shell_no_carry(E, lanes, dtype="float32"):
    """Shell with a runtime trip count and no carried value."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype), iters: T.int32):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            out_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    for _t in T.serial(iters):
                        out_ub[i] = x_ub[i] * T.float32(2.0)
            T.copy(out_ub, Y)

    return main


def _k_runtime_shell_carry(E, lanes, dtype="float32"):
    """Seed store + runtime-bound self-RMW: the carry pattern with a dynamic K."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype), iters: T.int32):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(0.0)
                    for _t in T.serial(iters):
                        y_ub[i] = y_ub[i] + x_ub[i]
            T.copy(y_ub, Y)

    return main


def _k_shell_no_carry(E, lanes, dtype="float32"):
    """Loop body only reads x and writes out: no carried value, no seed."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            out_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    for _t in T.serial(3):
                        out_ub[i] = x_ub[i] * T.float32(2.0)
            T.copy(out_ub, Y)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("E,lanes,tail", [(256, 64, False), (150, 128, True)])
def test_inside_serial_shell_is_preserved(E, lanes, tail):
    """The shell survives vectorization; for a non-divisible extent the tail
    path keeps its `remaining` counter, and the carrier stays in UB."""
    stages, error = _lower_capture(_k_newton_inside(E, lanes))
    assert error is None, f"the inside-serial form must lower, got: {error}"

    vec = str(stages["vectorize"])
    # No carry promotion any more: the carrier keeps round-tripping through UB
    # and the ordering comes from the backend mem_bar (verify writes the value).
    assert "_carry" not in vec, "the carry promotion is retired"
    assert _serial_shells_named(stages["vectorize"], "_t"), "the serial shell must survive vectorization"

    if tail:
        assert "remaining" in vec, "the tail path must keep its remaining counter"
    else:
        assert "y_ub" in vec, "the carrier must still be accessed by name"
        assert vec.count("vstore") >= 2, "the seed and the in-shell update both stay UB stores"
        # Legalize is identity for loops: still scalar, no VMI.
        legal = str(stages["legalize"])
        assert "vload" not in legal and "vstore" not in legal


@pytest.mark.pto
@pytest.mark.parametrize(
    "builder,carried",
    [(_k_runtime_shell_no_carry, False), (_k_runtime_shell_carry, True)],
)
def test_runtime_bound_shell_is_admitted_and_preserved(builder, carried):
    """The trip count may be a runtime scalar: the bound is copied through and
    the shell survives, with or without a UB carrier."""
    E, lanes = 256, 64
    stages, error = _lower_capture(builder(E, lanes))
    assert error is None, f"a runtime-bound shell must lower, got: {error}"

    shells = _serial_shells_named(stages["vectorize"], "_t")
    assert shells, "the runtime-bound shell must survive vectorization"
    assert not isinstance(shells[0].extent, tvm.tirx.IntImm), f"the trip count must stay a runtime expression, got {shells[0].extent}"
    vec = str(stages["vectorize"])
    assert "_carry" not in vec, "no carry pattern here: nothing to promote"


@contextmanager
def _ghost_extent_hook():
    """Rewrite the admitted serial shell's extent into a never-defined Var.

    The frontend cannot express an undefined variable, so the hook wraps
    ``VerifyParallelToPTO`` and rewrites the module right before the real pass
    runs. Only the shell bound changes; everything else stays as lowered.
    """
    ghost = tvm.tirx.Var("ghost_extent", "int32")
    original = ascend_transform.VerifyParallelToPTO

    def factory():
        real = original()

        def run(mod):
            def post(node):
                if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.SERIAL:
                    return tvm.tirx.For(
                        node.loop_var,
                        node.min,
                        ghost,
                        node.kind,
                        node.body,
                        node.thread_binding,
                        node.annotations,
                    )
                return None

            functions = {
                name: func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, post)) for name, func in mod.functions.items()
            }
            return real(tvm.IRModule(functions, mod.attrs))

        return run

    ascend_transform.VerifyParallelToPTO = factory
    try:
        yield
    finally:
        ascend_transform.VerifyParallelToPTO = original


@pytest.mark.pto
def test_undefined_variable_in_shell_bounds_is_rejected():
    """A shell bound that references a never-defined Var must not be admitted.

    The shell is copied through verbatim, so such a variable would survive into
    the output TIR; Verify runs the same reference check here as it does at every
    other expression position.
    """

    E, lanes = 256, 64
    with _ghost_extent_hook():
        _, error = _lower_capture(_k_shell_no_carry(E, lanes), expect_error=True)
    assert error is not None, "an undefined shell bound must be rejected"
    assert "never defined" in str(error), f"expected the undefined-variable diagnostic, got: {error}"


@contextmanager
def _opaque_extent_hook():
    """Replace the admitted serial shell's extent with an opaque extern call.

    The frontend cannot express such a bound, so the hook rewrites the module
    right before ``VerifyParallelToPTO`` runs. Everything else is untouched.
    """
    opaque = tvm.tirx.call_extern("int32", "pr11_trip_count")
    original = ascend_transform.VerifyParallelToPTO

    def factory():
        real = original()

        def run(mod):
            def post(node):
                if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.SERIAL:
                    return tvm.tirx.For(
                        node.loop_var,
                        node.min,
                        opaque,
                        node.kind,
                        node.body,
                        node.thread_binding,
                        node.annotations,
                    )
                return None

            functions = {
                name: func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, post)) for name, func in mod.functions.items()
            }
            return real(tvm.IRModule(functions, mod.attrs))

        return run

    ascend_transform.VerifyParallelToPTO = factory
    try:
        yield
    finally:
        ascend_transform.VerifyParallelToPTO = original


@pytest.mark.pto
def test_opaque_call_in_shell_bounds_is_rejected():
    """A bound that mentions no lane variable is not automatically supported.

    The shell is copied into the vectorized loop header, so an opaque call would
    run once per chunk instead of once per source iteration and neither its
    result nor its side effects are proven.
    """
    E, lanes = 256, 64
    with _opaque_extent_hook():
        _, error = _lower_capture(_k_shell_no_carry(E, lanes), expect_error=True)
    assert error is not None, "an opaque call as a shell bound must be rejected"
    assert "lane-uniform scalar" in str(error), f"expected the lane-uniform bound diagnostic, got: {error}"
