"""PTO Parallel control-flow tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``LegalizeParallelToPTO`` / the final lowering (no PTOAS/Bisheng device
compile, no device execution unless the case is explicitly a numeric one).

Expected contract:

* a lane-varying ``if`` / ``if``-``else`` under ``T.Parallel`` is rewritten by
  Legalize into the safe scalar ``Select`` form: stores are paired by written
  address, and a store without a counterpart in the other arm keeps the old
  value (``Select(cond, new, old_value)`` 闁?the same form a single-sided
  ``if`` produces);
* comparisons, ``T.min``/``T.max`` and boolean operators follow the same
  admission rules and end as ``vcmp``/``vsel`` (or are rejected with a
  diagnostic when the source semantics cannot be preserved);
* the frontend ``-T.infinity()`` sentinel is rewritten to a ``FloatImm``
  constant, so no frontend op name survives Legalize;
* explicit padding (``T.Parallel(P)`` with ``if i < E ... else ...``) keeps its
  ``else`` branch and legalizes to the Select form;
* Legalize itself never emits VMI and never widens a conditional load: the
  runtime address safety of full-width/masked loads stays the caller's
  contract, and a lowering that cannot preserve the source condition is
  rejected with a diagnostic.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tilelang
import tvm


@contextmanager
def _stage_capture():
    """Capture the modules around Legalize and after Vectorize.

    ``captured["legalize"]`` is the module LegalizeParallelToPTO returned
    (still scalar parallel TIR: Select form, no VMI yet);
    ``captured["vectorize"]`` is the module VectorizeParallelToPTO returned
    (the VMI form).
    """
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
    """Lower *func* for PTO and return (captured_stages, error)."""
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


def _legalize_direct(func):
    """Run Legalize directly so front-end Simplify cannot erase scalar Min/Max."""
    target = normalize_pto_target("pto")
    mod = tvm.IRModule.from_expr(func.with_attr("target", target))
    with target:
        return ascend_transform.LegalizeParallelToPTO()(mod)


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _collect(mod, predicate):
    found = []

    def visit(node):
        if predicate(node):
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


def _expr_loads(expr):
    """Buffer names read by an expression (no module wrapper)."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad):
            found.append(node.buffer.name)

    tvm.tirx.stmt_functor.post_order_visit(expr, visit)
    return set(found)


def _vmi_call_names(mod):
    names = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op = getattr(node, "op", None)
            name = getattr(op, "name", None)
            if isinstance(name, str) and name.startswith("tl.vmi."):
                names[name] = names.get(name, 0) + 1

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return names


def _selects(mod):
    return _collect(mod, lambda n: isinstance(n, tvm.tirx.Select))


def _if_then_elses(mod):
    return _collect(mod, lambda n: isinstance(n, tvm.tirx.IfThenElse))


def _max_min_nodes(mod):
    return _collect(
        mod,
        lambda n: isinstance(n, (tvm.tirx.Max, tvm.tirx.Min)),
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_if_else(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B_init: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i] * 2
                    else:
                        b[i] = a[i] + 1
            T.copy(b, B)

    return main


def _k_minmax(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        C: T.Tensor((P,), dtype),
        D: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            c = T.alloc_shared((P,), dtype)
            d = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c[i] = T.max(a[i], b[i])
                    d[i] = T.min(a[i], b[i])
            T.copy(c, C)
            T.copy(d, D)

    return main


def _k_explicit_padding(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(P):
                    if i < E:
                        b[i] = a[i] + a[i]
                    else:
                        b[i] = a[i]
            T.copy(b, B)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_if_else_legalizes_to_select_without_residual_branch():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_if_else(E, lanes))

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "Legalize must remove the lane-varying IfThenElse"
    assert _selects(legalized), "the both-arms store must become a Select"

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, "the comparison must lower to vcmp/vcmps"
    assert "tl.vmi.vsel" in names, "the Select must lower to vsel"
    assert "tl.vmi.vand" not in names, "this case must not need the boolean combine"

    # A conditional nested inside a branch recurses through the same rewrite
    # and ends as nested Selects (no residual IfThenElse anywhere).
    nested, _ = _lower_capture(_k_nested_three_way())
    nested_legalized = nested["legalize"]
    assert not _if_then_elses(nested_legalized), "nested conditionals must be legalized away"
    assert len(_selects(nested_legalized)) >= 2, "the nested predicates must become nested Selects"
    nested_names = _vmi_call_names(nested["vectorize"])
    assert nested_names.get("tl.vmi.vcmp", 0) >= 2, f"both comparisons must lower to vcmp, got {nested_names}"
    assert "tl.vmi.vsel" in nested_names, "the Selects must lower to vsel"


def _predicated_stores(stmt):
    return [n for n in _collect(stmt, lambda n: isinstance(n, tvm.tirx.BufferStore)) if n.predicate is not None]


def _stores(stmt):
    return _collect(stmt, lambda n: isinstance(n, tvm.tirx.BufferStore))


@pytest.mark.pto
def test_conditional_arm_that_reads_its_own_write_keeps_arm_order():
    """A store may read what an earlier store of the same arm wrote.

    Both arms write `y[0]` and `y[1]`, in opposite orders: the then arm writes
    `y[0]` and then reads it for `y[1]`, the else arm writes `y[1]` and then reads
    it for `y[0]`. The precedence mask keeps each arm's own order, so the arms
    cannot share positions; every store is emitted in its own position and pads
    the other side with a self-copy. The emitted order still keeps the then arm's
    `y[0]` before its `y[1]`, and no store is predicated."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), "float32"), Y: T.Tensor((2, P), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            y = T.alloc_shared((2, P), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        y[0, i] = a[i]
                        y[1, i] = y[0, i] * 2
                    else:
                        y[1, i] = a[i]
                        y[0, i] = y[1, i] * 3
            T.copy(y, Y)

    mod, error = _lower_capture(main, expect_error=True)
    assert error is None, f"the ordered arm read-after-write must lower, got: {error}"

    legalized = mod["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    # The two arms order their stores oppositely, so no position can carry both:
    # each store gets its own position, and what matters here is that the emitted
    # order still respects the dependency.
    assert len(stores) >= 2, f"expected at least one store per destination, got {len(stores)}"
    assert all(st.predicate is None for st in stores), "no store may stay predicated"
    order = [int(str(st.indices[0])) for st in stores]
    assert order.index(1) > order.index(0), f"the destinations must keep their own order, got row indices {order}"
    assert len(_selects(legalized)) >= 2, "every store needs its own value choice"


@pytest.mark.pto
def test_minmax_legalizes_without_residual_max_min_nodes():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_minmax(E, lanes))

    legalized = stages["legalize"]
    assert not _max_min_nodes(legalized), "Legalize must rewrite Min/Max into the Select form"
    assert _selects(legalized), "the rewritten Min/Max must be a Select before vectorization"

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vsel" in names, "the rewritten Min/Max must lower to vsel"


def _k_uniform_minmax_scalar(E=256, lanes=64):
    """`T.min` over two runtime int8 scalars, folded into a varying add."""

    @T.prim_func
    def main(
        M: T.int8,
        N: T.int8,
        A: T.Tensor((E,), "int8"),
        Y: T.Tensor((E,), "int8"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "int8")
            y = T.alloc_shared((E,), "int8")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c = T.min(M, N)
                    y[i] = a[i] + c * c
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_uniform_minmax_select_stays_scalar_in_the_unit():
    """Legalize rewrites Min/Max to Select even when both operands are
    lane-uniform runtime scalars, so the converted value must stay scalar:
    a fully lane-uniform Select is a valid PTODSL value (pto.select), and
    keeping it scalar preserves the scalar-domain arithmetic Verify
    admitted -- promoting it to a vreg would turn `c * c` into an int8
    vector multiply the VMI element set does not carry."""

    stages, error = _lower_capture(_k_uniform_minmax_scalar())
    assert error is None, f"the uniform Min/Max form must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel", 0) == 0, f"a fully lane-uniform Select must not become a vsel, got {names}"
    assert names.get("tl.vmi.vcmp", 0) == 0 and names.get("tl.vmi.vcmps", 0) == 0, (
        f"the uniform condition must fold as a scalar compare, got {names}"
    )
    # The scalar value reaches the varying add and is broadcast exactly at
    # that combination point (one vbrc for the scalar addend); the Select
    # itself survives as a scalar TIR node.
    assert names.get("tl.vmi.vbrc", 0) == 1, f"only the add's scalar operand should be broadcast, got {names}"
    assert _selects(stages["vectorize"]), "the scalar Select must survive"


@pytest.mark.pto
def test_explicit_padding_keeps_its_else_branch():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_explicit_padding(E, lanes))

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, "the guard must lower to a comparison"
    assert "tl.vmi.vsel" in names, "the preserved else must lower to vsel"
    assert "tl.vmi.vstore" in names, "the guarded store must survive as a masked vstore"
    # The guard covers [0, E) of the user's P domain: the vstore mask is the
    # full mask, never a remaining-based prefix mask.
    alloc_names = {node.buffer.name for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.AllocBuffer))}
    assert "remaining" not in alloc_names, f"explicit padding must not need a tail mask: {alloc_names}"


@pytest.mark.pto
def test_conditional_load_pre_evaluates_under_the_caller_contract():
    """A lane-varying guard around a load is admitted: Legalize keeps the load
    inside a Select arm (both arms pre-evaluated); the runtime readability of
    that full-width access is the input program's memory contract, not
    something the compiler proves."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if i < E - 1:
                        b[i] = a[i + 1]
                    else:
                        b[i] = a[i]
            T.copy(b, B)

    stages, _ = _lower_capture(main)
    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the guard must be legalized away"
    selects = _selects(legalized)
    assert selects, "the guarded load must be preserved inside a Select arm"

    loads = _collect(legalized, lambda n: isinstance(n, tvm.tirx.BufferLoad))
    assert loads, "the guarded load must survive (pre-evaluated arm), not be dropped"


@pytest.mark.pto
def test_divergent_branch_destinations_read_back_their_destination():
    """Arms that write different destinations each keep their own buffer's
    old value on the lanes the condition blocks: each destination gets its own
    read-back value choice (unpredicated store)."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B_init: T.Tensor((P,), "float32"),
        C_init: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
        C: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            c = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            T.copy(B_init, b)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i] * 2
                    else:
                        c[i] = a[i] + 1
            T.copy(b, B)
            T.copy(c, C)

    stages, error = _lower_capture(main)
    assert error is None, f"divergent destinations must lower: {error}"
    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"

    stores = _stores(legalized)
    assert len(stores) == 2, f"each destination gets its own store, got {len(stores)}"
    assert {s.buffer.name for s in stores} == {"b", "c"}
    for store in stores:
        # Each store keeps its own destination's old value: the value choice
        # reads that destination back, and the store stays unpredicated.
        assert store.predicate is None, f"store into {store.buffer.name} must not be predicated"
        assert store.buffer.name in _expr_loads(store.value), (
            f"the value choice for {store.buffer.name} must read it back, got {sorted(_expr_loads(store.value))}"
        )
    assert len(_selects(legalized)) >= 2, "each destination needs its own value choice"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel", 0) >= 2, f"each divergent destination must become a value choice, got {names}"
    assert "tl.vmi.vand" not in names, f"no mask combine without a predicated store, got {names}"
    assert names.get("tl.vmi.vstore", 0) == 2, f"each destination keeps one store, got {names}"


@pytest.mark.pto
def test_infinity_constant_legalizes_to_floatimm():
    """The frontend ``-T.infinity()`` sentinel is rewritten to a FloatImm
    constant by Legalize; Vectorize's constant handling covers it and no
    frontend op name survives into the vectorized IR."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i]
                    else:
                        b[i] = -T.infinity("float32")
            T.copy(b, B)

    stages, error = _lower_capture(main)
    assert error is None, f"infinity must lower as a constant: {error}"
    for stage in ("legalize", "vectorize"):
        text = _body_of(stages[stage]).__str__()
        assert "infinity" not in text, f"the frontend infinity op must not survive {stage}"
    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel") == 1, f"the Select must lower to vsel, got {names}"
    assert names.get("tl.vmi.vbrc", 0) >= 1, "the constant must reach a vbrc"


def _k_nested_three_way(E=150, lanes=128, dtype="float32"):
    """Three-way classify: nested if inside the then-arm, plain else."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        C: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            c = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        if b[i] > 0:
                            c[i] = a[i] * 2
                        else:
                            c[i] = b[i] * 2
                    else:
                        c[i] = a[i] + b[i]
            T.copy(c, C)

    return main


def _k_arm_read_then_write(E=150, lanes=128, dtype="float32"):
    """One arm reads a buffer and then overwrites it (WAR); the other writes
    and then reads it back (RAW). Both need the arm's own order preserved."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), dtype), Y: T.Tensor((2, P), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((2, P), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[0, i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[1, i] = y[0, i]
                        y[0, i] = T.float32(1.0)
                    else:
                        y[0, i] = T.float32(2.0)
                        y[1, i] = y[0, i]
            T.copy(y, Y)

    return main


def _k_war_asymmetric_arms(E=150, lanes=128, dtype="float32"):
    """A WAR inside the then arm, and the else arm writes fewer destinations."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), dtype), Y: T.Tensor((2, P), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((2, P), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[0, i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[1, i] = y[0, i]
                        y[0, i] = T.float32(1.0)
                    else:
                        y[0, i] = T.float32(2.0)
            T.copy(y, Y)

    return main


def _k_scalar_bool_select(E=150, lanes=128, dtype="float32"):
    """Both arms write one address and the condition is a bare scalar bool."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag:
                        y[i] = a[i]
                    else:
                        y[i] = b[i]
            T.copy(y, Y)

    return main


def _k_scalar_bool_predicate(E=150, lanes=128, dtype="float32"):
    """One-sided branch under a bare scalar bool."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag:
                        y[i] = a[i]
            T.copy(y, Y)

    return main


def _k_scalar_bool_and_lane_cond(E=150, lanes=128, dtype="float32"):
    """A scalar bool combined with a lane-varying condition."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag and a[i] > T.float32(0.0):
                        y[i] = a[i]
            T.copy(y, Y)

    return main


def _k_boolean_mask(E, lanes, dtype="float32"):
    """A boolean expression over two comparisons used as the guard; the mask
    is bound first (a value position), so the condition keeps the mask-logic
    form through Legalize and Vectorize converts it to vand."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((E,), dtype),
        C: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            b = T.alloc_shared((E,), dtype)
            c = T.alloc_shared((E,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    m = (a[i] > 0) & (b[i] > 0)
                    if m:
                        c[i] = a[i]
                    else:
                        c[i] = b[i]
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_conditional_arm_with_read_then_write_keeps_arm_order():
    """An arm that reads a buffer and then overwrites it (WAR) folds without
    losing either access.

    Both arms touch `y[0]` in conflicting orders: the then arm reads it and then
    clobbers it, while the else arm writes it and then reads it back. The
    precedence mask keeps each arm's own order, so the DP cannot place those two
    stores in one position; it emits the conflicting destination once per arm
    instead, filling the other side with a self-copy that leaves the lane's value
    untouched (one extra position, never a reordering)."""

    stages, error = _lower_capture(_k_arm_read_then_write())
    assert error is None, f"WAR/RAW inside an arm must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    # Init store + three value choices (the conflicting destination is emitted
    # once per arm): four stores in total.
    assert len(stores) == 4, f"expected init + three value choices, got {len(stores)}"
    choices = [st for st in stores if isinstance(st.value, tvm.tirx.Select)]
    assert len(choices) == 3, f"expected three value choices, got {len(choices)}"
    assert all(st.predicate is None for st in stores), "no store may stay predicated"


@pytest.mark.pto
def test_war_with_unequal_arms_pairs_by_address():
    """A WAR inside the then arm must not force the arms apart when the else arm
    writes fewer destinations: the reorder step already found an emission order
    that keeps the clobbered read first, so the destinations pair by address
    instead of splitting into per-store read-backs."""

    stages, error = _lower_capture(_k_war_asymmetric_arms())
    assert error is None, f"WAR inside an arm must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    # One init store plus one value choice per destination: y1 keeps its own
    # value on the else side, y0 pairs across the arms.
    assert len(stores) == 3, f"expected init + two aligned positions, got {len(stores)}"
    choices = [st for st in stores if isinstance(st.value, tvm.tirx.Select)]
    assert len(choices) == 2, f"expected two value choices, got {len(choices)}"
    assert all(st.predicate is None for st in stores), "no store may stay predicated"


@pytest.mark.pto
@pytest.mark.parametrize(
    "builder,one_sided",
    [(_k_scalar_bool_select, False), (_k_scalar_bool_predicate, True)],
)
def test_scalar_bool_condition_becomes_a_full_mask(builder, one_sided):
    """A bare scalar bool has no operand to compare, so it becomes the active
    lane count of a second `create_mask`: all lanes for true, none for false.
    The result is an all-true / all-false lane mask that `vsel` can consume,
    both as a two-sided value choice and as a one-sided branch predicate."""

    stages, error = _lower_capture(builder())
    assert error is None, f"a scalar bool condition must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    assert _selects(legalized), "the branch must become a value choice"
    if one_sided:
        stores = _stores(legalized)
        assert len(stores) == 1, f"expected one store, got {len(stores)}"
        assert stores[0].predicate is None, "the read-back store must not be predicated"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.create_mask", 0) >= 2, f"the scalar bool must produce its own create_mask, got {names}"
    assert "tl.vmi.vcmps" not in names, f"a scalar bool needs no vector compare carrier, got {names}"
    assert "tl.vmi.vsel" in names, f"the Select must lower to vsel, got {names}"


@pytest.mark.pto
def test_scalar_bool_in_a_conjunction_is_widened_before_the_combine():
    """`flag and (a[i] > 0)` is distributed into nested keep-value choices: the
    scalar side becomes its own `create_mask` first, and both masks are
    conjoined into the predicated store. Only the lane comparison needs a
    vector compare."""

    stages, error = _lower_capture(_k_scalar_bool_and_lane_cond())
    assert error is None, f"a scalar bool in a conjunction must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.create_mask", 0) >= 2, f"the scalar bool must be widened first, got {names}"
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, f"the lane comparison still needs a vector compare, got {names}"
    assert "tl.vmi.vsel" in names, f"the conjunction must lower to a value choice, got {names}"


@pytest.mark.pto
def test_boolean_mask_value_lowers_to_vand():
    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_boolean_mask(E, lanes))

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vand") == 1, f"vand must be emitted once, got {names}"
    assert "tl.vmi.vsel" in names, "the guarded Select must lower to vsel"
    assert names.get("tl.vmi.vcmp", 0) >= 2, "both comparisons must lower to vcmp"

    # The boolean combine is the existing PTODSL bitwise vector op on mask
    # registers, so the emitted PTODSL name equals the TileLang op name.
    kernel = tilelang.compile(_k_boolean_mask(E, lanes), target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    assert "pto.vmi.vand(" in source, "the boolean combine must emit pto.vmi.vand"


@pytest.mark.pto
def test_lane_uniform_store_is_still_rejected():
    """A store whose address does not vary with the lane stays out of scope:
    Verify rejects it and Legalize never materialises a reducer for it."""

    E, lanes = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            b = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if i < E // 2:
                        b[0] = a[i]
            T.copy(b, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a lane-uniform store must stay rejected at this stage"
    message = str(error)
    assert "lane-uniform" in message, f"expected the lane-uniform rejection, got: {message}"


def _k_one_sided_nested(E=150, lanes=128, dtype="float32"):
    """A one-sided branch nested in the then arm of another one-sided branch."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), dtype), B: T.Tensor((P,), dtype), Y: T.Tensor((2, P), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((2, P), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[0, i] = a[i]
                        if a[i] > T.float32(1.0):
                            y[1, i] = b[i]
            T.copy(y, Y)

    return main


def _k_one_sided_with_sibling_write(E=150, lanes=128, dtype="float32"):
    """The destination of a one-sided branch is also written unconditionally."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), dtype), B: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[i] = a[i]
                    if a[i] > T.float32(0.0):
                        y[i] = b[i]
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_one_sided_nested_branch_folds_once_per_destination():
    """Two destinations, each folded exactly once: the read-back is synthesised
    at the emit point, so nesting does not duplicate a destination's store."""

    stages, error = _lower_capture(_k_one_sided_nested())
    assert error is None, f"nested one-sided branches must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "both conditionals must be folded"
    stores = _stores(legalized)
    assert len(stores) == 2, f"expected one store per destination, got {len(stores)}"
    assert all(isinstance(st.value, tvm.tirx.Select) for st in stores), (
        "each one-sided destination must keep its entry value through one Select"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel", 0) >= 2, f"both one-sided destinations must become value choices, got {names}"
    assert "tl.vmi.vand" not in names, f"no mask combine without a predicated store, got {names}"
    assert names.get("tl.vmi.vstore", 0) == 2, f"each destination keeps one store, got {names}"


@pytest.mark.pto
def test_one_sided_branch_keeps_a_single_item_per_destination():
    """A sibling statement writing the same destination must not create a second
    canonical item: the branch still folds into exactly one value choice."""

    stages, error = _lower_capture(_k_one_sided_with_sibling_write())
    assert error is None, f"a one-sided branch must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be folded"
    stores = _stores(legalized)
    # The sibling write plus one folded value choice for the same destination.
    assert len(stores) == 2, f"expected two stores, got {len(stores)}"
    selects = [st for st in stores if isinstance(st.value, tvm.tirx.Select)]
    assert len(selects) == 1, f"expected exactly one value choice, got {len(selects)}"


# ---------------------------------------------------------------------------
# Conditional destination addresses
#
# A conditional store becomes one arm of a value choice, so Legalize pairs the
# two arms by destination address. That only works for the canonical per-lane
# form (leading indices constant, the T.Parallel variable last), and Verify
# enforces it up front: the same shape restriction exists in Verify and Legalize
# instead of failing late with a misleading pairing error.
#
# Plain (unconditional) stores keep the wider contract: any address the shared
# analysis proves continuous is admitted, offsets included.
# ---------------------------------------------------------------------------


def _k_conditional_interleaved_destination(E=150, lanes=128, dtype="float32"):
    """A conditional store whose destination interleaves lanes."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        Y: T.Tensor((2 * P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((2 * P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        y[2 * i] = a[i]
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_conditional_interleaved_destination_is_rejected_at_verify():
    """An interleaved destination cannot be paired either; Verify rejects it."""

    _, error = _lower_capture(_k_conditional_interleaved_destination(), expect_error=True)
    assert error is not None, "an interleaved destination must be rejected"
    # The access check runs before the pairing contract, so an interleaved
    # address is reported as a non-continuous one (the more precise diagnostic).
    message = str(error)
    assert "continuous" in message or "cannot pair" in message, f"expected an address diagnostic, got: {message}"


def _k_plain_offset_destination(E=150, lanes=128, dtype="float32"):
    """An unconditional store with a constant, 32-byte aligned offset."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        Y: T.Tensor((P + 8,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P + 8,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[i + 8] = a[i]
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_plain_offset_destination_stays_supported():
    """The tightening is limited to conditional destinations: an unconditional
    store may still use a continuous address with an offset."""

    stages, error = _lower_capture(_k_plain_offset_destination())
    assert error is None, f"a plain offset destination must lower, got: {error}"

    assert not _if_then_elses(stages["legalize"]), "there is no branch to rewrite"
    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vstore", 0) == 1, f"expected one store, got {names}"


@pytest.mark.pto
def test_2d_outer_vectorized_conditional_write_folds():
    """A one-sided conditional write in a 2D unit whose *outer* dimension is
    the vectorized one. Verify accepts the canonical form (IntImm leading
    indices, the vectorized variable last), so Legalize must fold the store
    against the layout-selected lane variable -- not against the syntactically
    innermost loop of the nest, which is the kept coordinate here."""

    E, F, lanes = 128, 4, 64

    @T.prim_func
    def main(A: T.Tensor((F, E), "int32"), C: T.Tensor((F, E), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((F, E), "int32")
            c = T.alloc_shared((F, E), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(E, F):
                    if a[j, i] > 0:
                        c[0, i] = a[j, i] + i
            T.copy(c, C)

    stages, error = _lower_capture(main)
    assert error is None, f"the 2D outer-vectorized conditional must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected the folded store, got {len(stores)}"
    assert stores[0].predicate is None, "the folded store must stay unpredicated"
    assert _selects(legalized), "the arm must become a value choice"
    assert "c" in _expr_loads(stores[0].value), (
        f"the missing side must read the destination's old value back, got {sorted(_expr_loads(stores[0].value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, f"the value choice must lower to vsel, got {names}"


@pytest.mark.pto
def test_2d_inner_vectorized_conditional_write_still_folds():
    """The inner-vectorized counterpart keeps folding: there the selected lane
    variable coincides with the syntactically innermost loop, so the default
    path must be untouched by the lane-variable fix."""

    E, F, lanes = 128, 4, 64

    @T.prim_func
    def main(A: T.Tensor((E, F), "int32"), C: T.Tensor((E, F), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((E, F), "int32")
            c = T.alloc_shared((E, F), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for _i, j in T.Parallel(E, F):
                    if a[0, j] > 0:
                        c[0, j] = a[0, j] + j
            T.copy(c, C)

    stages, error = _lower_capture(main)
    assert error is None, f"the 2D inner-vectorized conditional must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected the folded store, got {len(stores)}"
    assert stores[0].predicate is None, "the folded store must stay unpredicated"
    assert _selects(legalized), "the arm must become a value choice"
    assert "c" in _expr_loads(stores[0].value), (
        f"the missing side must read the destination's old value back, got {sorted(_expr_loads(stores[0].value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, f"the value choice must lower to vsel, got {names}"


@pytest.mark.pto
def test_conditional_offset_destination_folds():
    """A one-sided conditional write to a constant-offset destination
    (`y[i + 8]`) folds like the canonical form: the offset keeps the access
    per-lane continuous, the pairing and the pad read-back stay well-defined,
    and the shifted window's 32-byte alignment is codegen legality's contract
    (VerifyPTOCodegenLegality), not the value-choice rewrite's."""

    E, lanes = 128, 64

    @T.prim_func
    def main(A: T.Tensor((E,), "int32"), Y: T.Tensor((E + 8,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "int32")
            y = T.alloc_shared((E + 8,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        y[i + 8] = a[i] + i
            T.copy(y, Y)

    stages, error = _lower_capture(main)
    assert error is None, f"the offset conditional must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected the folded store, got {len(stores)}"
    assert stores[0].predicate is None, "the folded store must stay unpredicated"
    assert _selects(legalized), "the arm must become a value choice"
    assert "y" in _expr_loads(stores[0].value), (
        f"the missing side must read the destination's old value back, got {sorted(_expr_loads(stores[0].value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, f"the value choice must lower to vsel, got {names}"


def _k_2d_kept_row_conditional_inner(E=128, F=4, lanes=64):
    """2D inner-vectorized: the kept row coordinate is the leading index."""

    @T.prim_func
    def main(A: T.Tensor((F, E), "int32"), C: T.Tensor((F, E), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((F, E), "int32")
            c = T.alloc_shared((F, E), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(F, E):
                    if a[i, j] > 0:
                        c[i, j] = a[i, j] + 1
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_2d_inner_vectorized_row_coordinate_conditional_write_folds():
    """A 2D unit whose inner dimension is vectorized and whose conditional
    destination uses the kept row coordinate as the leading index (`c[i, j]`).
    The leading index is uniform over the chunk and the address stays
    `start + lane` per row -- VerifyAccess proves that continuity before the
    pairing rule runs, so the fold must not require an IntImm there."""

    stages, error = _lower_capture(_k_2d_kept_row_conditional_inner())
    assert error is None, f"the row-coordinate conditional must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected the folded store, got {len(stores)}"
    assert stores[0].predicate is None, "the folded store must stay unpredicated"
    assert _selects(legalized), "the arm must become a value choice"
    assert "c" in _expr_loads(stores[0].value), (
        f"the missing side must read the destination's old value back, got {sorted(_expr_loads(stores[0].value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, f"the value choice must lower to vsel, got {names}"


def _k_2d_kept_row_conditional_outer(E=128, F=4, lanes=64):
    """2D outer-vectorized: the kept inner coordinate is the leading index."""

    @T.prim_func
    def main(A: T.Tensor((F, E), "int32"), C: T.Tensor((F, E), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((F, E), "int32")
            c = T.alloc_shared((F, E), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(E, F):
                    if a[j, i] > 0:
                        c[j, i] = a[j, i] + 1
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_2d_outer_vectorized_row_coordinate_conditional_write_folds():
    """The outer-vectorized counterpart (`c[j, i]`, vectorized outer variable
    last): the kept inner coordinate leads -- the same uniform-leading form
    seen from the other side of the layout choice."""

    stages, error = _lower_capture(_k_2d_kept_row_conditional_outer())
    assert error is None, f"the kept-coordinate conditional must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected the folded store, got {len(stores)}"
    assert stores[0].predicate is None, "the folded store must stay unpredicated"
    assert _selects(legalized), "the arm must become a value choice"
    assert "c" in _expr_loads(stores[0].value), (
        f"the missing side must read the destination's old value back, got {sorted(_expr_loads(stores[0].value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, f"the value choice must lower to vsel, got {names}"


def _k_per_lane_scalar_in_unit(E=150, lanes=64, dtype="float32"):
    """A per-lane scalar (`T.alloc_var`) used inside a T.Parallel unit."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        S: T.Tensor((P,), dtype),
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(S, s)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    t = T.alloc_var(dtype)
                    t = s[i]
                    y[i] = t
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_per_lane_scalar_inside_a_unit_is_rejected():
    """`T.alloc_var` has no promotion path here: a `local.var` access inside a
    converting unit is rejected before Legalize sees it."""

    _, error = _lower_capture(_k_per_lane_scalar_in_unit(), expect_error=True)
    assert error is not None, "a per-lane scalar must be rejected"
    assert "only UB (shared/shared.dyn) accesses are supported" in str(error), f"expected the scope diagnostic, got: {error}"


# ---------------------------------------------------------------------------
# Conditions that read a destination written by the same conditional
# ---------------------------------------------------------------------------


def _condition_snapshots(mod):
    return _collect(
        mod,
        lambda n: isinstance(n, tvm.tirx.Bind) and n.var.name.startswith("cond_snapshot_"),
    )


def _mentions(expr, var):
    """True when *var* occurs anywhere in *expr* (a nested Select included)."""
    found = []
    tvm.tirx.stmt_functor.post_order_visit(
        tvm.tirx.Evaluate(expr),
        lambda node: found.append(isinstance(node, tvm.tirx.Var) and node.same_as(var)),
    )
    return any(found)


def _ordered_plan_nodes(mod):
    """Return Bind/BufferStore nodes in their emitted statement order."""
    ordered = []

    def visit(stmt):
        if isinstance(stmt, tvm.tirx.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, (tvm.tirx.For, tvm.tirx.SBlock)):
            visit(stmt.body)
        elif isinstance(stmt, tvm.tirx.SBlockRealize):
            visit(stmt.block)
        elif isinstance(stmt, tvm.tirx.AttrStmt):
            visit(stmt.body)
        elif isinstance(stmt, (tvm.tirx.Bind, tvm.tirx.BufferStore)):
            ordered.append(stmt)

    visit(_body_of(mod))
    return ordered


def _k_single_in_place(E=256, lanes=64):
    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        a[i] = T.float32(0.0)
            T.copy(a, Y)

    return main


@pytest.mark.pto
def test_single_in_place_write_does_not_need_a_condition_snapshot():
    stages, error = _lower_capture(_k_single_in_place())
    assert error is None, f"a single in-place write must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized)
    assert not _condition_snapshots(legalized), "one final store cannot re-evaluate the condition after another write"
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected one folded store, got {stores}"
    assert isinstance(stores[0].value, tvm.tirx.Select)
    assert "a" in _expr_loads(stores[0].value), "the missing side must keep the destination's old value"
    assert "tl.vmi.vsel" in _vmi_call_names(stages["vectorize"])


def _k_multi_in_place(E=256, lanes=64):
    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            b = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = T.float32(7.0)
                    if a[i] > T.float32(0.0):
                        a[i] = T.float32(0.0)
                        y[i] = T.float32(1.0)
                    else:
                        a[i] = T.float32(-1.0)
                        y[i] = T.float32(2.0)
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_multi_in_place_writes_share_one_entry_condition_snapshot():
    stages, error = _lower_capture(_k_multi_in_place())
    assert error is None, f"multi in-place writes must lower, got: {error}"

    legalized = stages["legalize"]
    snapshots = _condition_snapshots(legalized)
    assert len(snapshots) == 1, f"expected one entry snapshot, got {snapshots}"
    condition = snapshots[0].var
    selects = _selects(legalized)
    assert len(selects) == 2, f"both destinations need a value choice: {selects}"
    for select in selects:
        assert select.condition.same_as(condition), "every store must use the entry condition, not re-read A after a write"

    ordered = _ordered_plan_nodes(legalized)
    snapshot_index = next(i for i, node in enumerate(ordered) if isinstance(node, tvm.tirx.Bind) and node.var.same_as(condition))
    assert snapshot_index > 0, "the condition snapshot must precede both stores"
    assert all(
        not isinstance(node, tvm.tirx.BufferStore) or not isinstance(node.value, tvm.tirx.Select) for node in ordered[:snapshot_index]
    ), "no value-choice store may execute before the entry condition"


def _k_one_sided_multi_in_place(E=256, lanes=64):
    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
        Z: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            z = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        a[i] = T.float32(0.0)
                        z[i] = T.float32(1.0)
            T.copy(a, Y)
            T.copy(z, Z)

    return main


@pytest.mark.pto
def test_one_sided_multi_in_place_shares_one_entry_condition_snapshot():
    """One arm that updates its own condition input twice still snapshots once.

    The acceptance case "the condition reads A, the branch rewrites A and then
    writes B" without an ``else``: both stores must reuse the entry condition,
    and the missing side reads the destination back.
    """
    stages, error = _lower_capture(_k_one_sided_multi_in_place())
    assert error is None, f"a one-sided in-place update pair must lower, got: {error}"

    legalized = stages["legalize"]
    snapshots = _condition_snapshots(legalized)
    assert len(snapshots) == 1, f"expected one entry snapshot, got {snapshots}"
    condition = snapshots[0].var
    selects = _selects(legalized)
    assert len(selects) == 2, f"both destinations need a value choice: {selects}"
    for select in selects:
        assert select.condition.same_as(condition), "the second store must reuse the entry condition, not re-read A"
        # A one-sided arm keeps the destination's old value on the false side.
        assert isinstance(select.false_value, tvm.tirx.BufferLoad), "the false side of a one-sided value choice reads the destination back"


def _k_nested_condition_after_write(E=256, lanes=64):
    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        C: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            b = T.alloc_shared((E,), "float32")
            c = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        b[i] = a[i] - T.float32(1.0)
                        if b[i] > T.float32(0.0):
                            b[i] = T.float32(0.0)
                            c[i] = T.float32(1.0)
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_nested_condition_snapshot_stays_after_the_preceding_write():
    stages, error = _lower_capture(_k_nested_condition_after_write())
    assert error is None, f"a nested in-place condition must lower, got: {error}"

    legalized = stages["legalize"]
    snapshots = _condition_snapshots(legalized)
    assert len(snapshots) == 1, f"only the inner condition needs a snapshot: {snapshots}"
    ordered = _ordered_plan_nodes(legalized)
    snapshot_index = next(
        i for i, node in enumerate(ordered) if isinstance(node, tvm.tirx.Bind) and node.var.name.startswith("cond_snapshot_")
    )
    assert snapshot_index > 0
    preceding = ordered[snapshot_index - 1]
    assert isinstance(preceding, tvm.tirx.BufferStore)
    assert preceding.buffer.name == "b", (
        "the inner condition must be evaluated after B is written, not hoisted to the outer condition's entry"
    )
    tail = [node for node in ordered[snapshot_index + 1 :] if isinstance(node, tvm.tirx.BufferStore)]
    assert tail, "the inner writes must follow the condition snapshot"
    for node in tail:
        # The inner writes consume the snapshot; an enclosing conditional wraps
        # them again with its own condition, so the snapshot is not necessarily
        # the outermost Select condition.
        assert _mentions(node.value, snapshots[0].var), "every inner write must consume the entry condition snapshot"


def _k_serial_condition(E=256, lanes=64):
    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(X, x)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    for k in T.serial(3):  # noqa: B007 - the name is asserted below
                        if x[i] > T.float32(0.0):
                            x[i] = x[i] + T.float32(1.0)
                            y[i] = x[i]
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_condition_snapshot_stays_inside_the_serial_shell():
    stages, error = _lower_capture(_k_serial_condition())
    assert error is None, f"a serial in-place condition must lower, got: {error}"

    legalized = stages["legalize"]
    snapshots = _condition_snapshots(legalized)
    assert snapshots, "the multiple writes need a condition snapshot"
    shells = _collect(
        legalized,
        lambda n: isinstance(n, tvm.tirx.For) and n.kind == tvm.tirx.ForKind.SERIAL and n.loop_var.name == "k",
    )
    assert shells, "the source serial shell must survive"
    for snapshot in snapshots:
        contained = []
        for shell in shells:
            tvm.tirx.stmt_functor.post_order_visit(
                shell.body,
                lambda node, snap=snapshot, acc=contained: acc.append(node.same_as(snap)),
            )
        assert any(contained), "a serial condition must be evaluated inside every iteration"


def _k_multi_without_condition_conflict(E=256, lanes=64):
    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((E,), "float32")
            a = T.alloc_shared((E,), "float32")
            b = T.alloc_shared((E,), "float32")
            T.copy(X, x)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if x[i] > T.float32(0.0):
                        a[i] = T.float32(0.0)
                        b[i] = T.float32(1.0)
            T.copy(b, Y)

    return main


@pytest.mark.pto
def test_multi_writes_with_unchanged_condition_input_add_no_snapshot():
    stages, error = _lower_capture(_k_multi_without_condition_conflict())
    assert error is None, f"an independent condition must lower, got: {error}"
    assert not _condition_snapshots(stages["legalize"]), "two Selects over an unchanged input may share the original expression"


# ---------------------------------------------------------------------------
# Uniform boolean combinations: the scalar domain Verify admitted survives
# ---------------------------------------------------------------------------

_BOOL_COMBINE = {
    "and": lambda p, q: (p > 0) & (q > 0),
    "or": lambda p, q: (p > 0) | (q > 0),
    "eq": lambda p, q: (p > 0) == (q > 0),
    "not": lambda p, q: ~(p > 0),
}


def _k_uniform_bool_select(op, E=256, lanes=64):
    """A fully lane-uniform boolean condition selecting two int8 scalars."""

    combine = _BOOL_COMBINE[op]

    @T.prim_func
    def main(
        P: T.int32,
        Q: T.int32,
        M: T.int8,
        N: T.int8,
        A: T.Tensor((E,), "int8"),
        Y: T.Tensor((E,), "int8"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "int8")
            y = T.alloc_shared((E,), "int8")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c = tvm.tirx.Select(combine(P, Q), M, N)
                    y[i] = a[i] + c * c
            T.copy(y, Y)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("op", ["and", "or", "eq", "not"])
def test_uniform_boolean_combination_stays_scalar_in_the_unit(op):
    """`(P>0) <op> (Q>0)` over runtime scalars must not build a lane mask.

    Forcing the mask promotes the selected int8 scalars into a vreg, and
    `c * c` then becomes an int8 `vmul` the VMI element set does not carry.
    """
    stages, error = _lower_capture(_k_uniform_bool_select(op))
    assert error is None, f"a fully lane-uniform `{op}` must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel", 0) == 0, f"scalar Select must not become vsel: {names}"
    assert names.get("tl.vmi.vcmp", 0) == 0 and names.get("tl.vmi.vcmps", 0) == 0, (
        f"the uniform condition must stay a scalar compare: {names}"
    )
    for mask_op in ("tl.vmi.vand", "tl.vmi.vor", "tl.vmi.vnot"):
        assert names.get(mask_op, 0) == 0, f"a fully lane-uniform combination must not lift to {mask_op}: {names}"
    assert names.get("tl.vmi.vmul", 0) == 0, f"the int8 scalar multiply must not become a vmul: {names}"
    assert _selects(stages["vectorize"]), "the scalar Select must survive"


def _k_varying_bool_select(E=256, lanes=64):
    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[i] = tvm.tirx.Select(
                        (a[i] > T.float32(0.0)) & (a[i] < T.float32(8.0)),
                        T.float32(1.0),
                        T.float32(0.0),
                    )
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_lane_varying_boolean_combination_still_uses_a_mask():
    """The counterpart: a lane-varying `&` keeps the mask lowering."""
    stages, error = _lower_capture(_k_varying_bool_select())
    assert error is None, f"a lane-varying `&` must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vand", 0) >= 1, f"the varying combine must use vand: {names}"
    assert names.get("tl.vmi.vsel", 0) >= 1, f"the varying Select must use vsel: {names}"


# ---------------------------------------------------------------------------
# Access predicates: absent or provably true only
# ---------------------------------------------------------------------------


def _k_plain_two_stores(E=256, lanes=64):
    @T.prim_func
    def main(
        P: T.int32,
        A: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] + T.float32(1.0)
                    y[i] = a[i] * T.float32(2.0)
            T.copy(y, Y)

    return main


def _with_store_predicate(func, factory):
    """Attach `factory(func)` as the predicate of every BufferStore."""

    def post(node):
        if isinstance(node, tvm.tirx.BufferStore):
            return tvm.tirx.BufferStore(node.buffer, node.value, node.indices, factory(func))
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.BufferStore"])
    return func.with_body(body)


@pytest.mark.pto
def test_provably_true_store_predicate_is_admitted_and_cleared():
    """A literal-true predicate limits nothing: Verify admits it, Legalize clears it."""
    stages, error = _lower_capture(_with_store_predicate(_k_plain_two_stores(), lambda f: tvm.tirx.const(True, "bool")))
    assert error is None, f"a provably-true predicate must be admitted, got: {error}"

    legalized = stages["legalize"]
    predicated = [node for node in _collect(legalized, lambda n: isinstance(n, tvm.tirx.BufferStore)) if node.predicate is not None]
    assert not predicated, "Legalize must clear an admitted predicate before folding"


@pytest.mark.pto
def test_loop_range_true_store_predicate_is_admitted_and_cleared():
    """`i < E` is provable from the unit's own range, not from a guess."""
    state = {}

    def pre(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.PARALLEL and "loop_var" not in state:
            state["loop_var"] = node.loop_var
        return None

    func = _k_plain_two_stores()

    def factory(f):
        return tvm.tirx.LT(state["loop_var"], tvm.tirx.IntImm("int32", 256))

    def post(node):
        if isinstance(node, tvm.tirx.BufferStore):
            return tvm.tirx.BufferStore(node.buffer, node.value, node.indices, factory(func))
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, pre, post, ["tirx.For", "tirx.BufferStore"])
    stages, error = _lower_capture(func.with_body(body))
    assert error is None, f"a range-provable predicate must be admitted, got: {error}"
    predicated = [
        node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferStore)) if node.predicate is not None
    ]
    assert not predicated, "Legalize must clear a range-provable predicate"


@pytest.mark.pto
def test_false_store_predicate_is_rejected():
    _, error = _lower_capture(
        _with_store_predicate(_k_plain_two_stores(), lambda f: tvm.tirx.const(False, "bool")),
        expect_error=True,
    )
    assert error is not None, "a false predicate must be rejected"
    assert "not provably true" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_runtime_store_predicate_is_rejected():
    _, error = _lower_capture(
        _with_store_predicate(_k_plain_two_stores(), lambda f: f.params[0] > tvm.tirx.IntImm("int32", 0)),
        expect_error=True,
    )
    assert error is not None, "a runtime predicate must be rejected"
    assert "not provably true" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_predicate_with_an_undefined_variable_is_rejected():
    """The reference check must run on the original expression.

    Simplifying first would fold `True | ghost` to True and hide the undefined
    variable.
    """
    ghost = tvm.tirx.Var("ghost_flag", "bool")

    def factory(f):
        return tvm.tirx.Or(tvm.tirx.const(True, "bool"), ghost)

    _, error = _lower_capture(_with_store_predicate(_k_plain_two_stores(), factory), expect_error=True)
    assert error is not None, "an undefined variable in a predicate must be rejected"
    # AdmitPredicates now owns the reference check, so the diagnostic names the
    # region scope rather than the unit's own "never defined" wording.
    assert "not defined in the lexical scope" in str(error), f"unexpected diagnostic: {error}"


# ---------------------------------------------------------------------------
# Handle-Bind views: rejected, not resolved
# ---------------------------------------------------------------------------


def _view_over_bound_pointer(func, tag="alias"):
    """Rewrite one unit store to write through a view of a Bind-produced pointer.

    Emits the shape `q = A.data; DeclBuffer(view, data=q)` the alias admission
    used to accept. The Bind is spliced into the enclosing SeqStmt so it
    dominates the rewritten store.
    """
    state = {}

    def post(node):
        if isinstance(node, tvm.tirx.BufferStore) and "view" not in state:
            src = node.buffer
            pointer = tvm.tirx.Var(
                tag + "_ptr",
                tvm.ir.PointerType(tvm.ir.PrimType(src.dtype), src.scope()),
            )
            view = tvm.tirx.decl_buffer(src.shape, src.dtype, data=pointer, scope=src.scope())
            state["view"] = view
            state["bind"] = tvm.tirx.Bind(pointer, src.data)
            return tvm.tirx.BufferStore(view, node.value, node.indices)
        if isinstance(node, tvm.tirx.SeqStmt) and "bind" in state and "spliced" not in state:
            state["spliced"] = True
            return tvm.tirx.SeqStmt([state["bind"]] + list(node.seq))
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.BufferStore", "tirx.SeqStmt"])
    return func.with_body(body)


@pytest.mark.pto
def test_buffer_view_over_a_bound_data_pointer_is_rejected():
    """A view the lane analysis cannot follow is refused at Verify.

    Ordering cannot repair an access that should never be admitted: the pointer
    may be external, a reinterpret cast, or a laundered copy, so its storage is
    unknown rather than aliased.
    """
    _, error = _lower_capture(_view_over_bound_pointer(_k_plain_two_stores()), expect_error=True)
    assert error is not None, "a view over a Bind-produced pointer must be rejected"
    assert "bound by a handle Bind" in str(error), f"unexpected diagnostic: {error}"


def _k_uniform_bool_nodes(op, E=256, lanes=64):
    """The node spellings (`And`/`Or`/`Not`) rather than the `bitwise_*` calls."""

    combine = {
        "and": lambda p, q: tvm.tirx.And(p > 0, q > 0),
        "or": lambda p, q: tvm.tirx.Or(p > 0, q > 0),
        "not": lambda p, q: tvm.tirx.Not(p > 0),
        "ne": lambda p, q: tvm.tirx.NE(p > 0, q > 0),
    }[op]

    @T.prim_func
    def main(
        P: T.int32,
        Q: T.int32,
        M: T.int8,
        N: T.int8,
        A: T.Tensor((E,), "int8"),
        Y: T.Tensor((E,), "int8"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "int8")
            y = T.alloc_shared((E,), "int8")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c = tvm.tirx.Select(combine(P, Q), M, N)
                    y[i] = a[i] + c * c
            T.copy(y, Y)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("op", ["and", "or", "not", "ne"])
def test_uniform_bool_node_spellings_stay_scalar(op):
    """`And`/`Or`/`Not`/`NE` nodes take the same scalar path as `bitwise_*`."""
    stages, error = _lower_capture(_k_uniform_bool_nodes(op))
    assert error is None, f"a fully lane-uniform `{op}` node must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel", 0) == 0, f"scalar Select must not become vsel: {names}"
    for mask_op in ("tl.vmi.vand", "tl.vmi.vor", "tl.vmi.vnot", "tl.vmi.vcmp"):
        assert names.get(mask_op, 0) == 0, f"`{op}` must not lift to {mask_op}: {names}"
    assert names.get("tl.vmi.vmul", 0) == 0, f"int8 multiply must not become vmul: {names}"


def _k_mask_with_scalar_eq(E=256, lanes=64):
    """A bool EQ/NE between a lane mask and a scalar bool keeps the truth table."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32"), FLAG: T.bool):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[i] = tvm.tirx.Select(
                        tvm.tirx.EQ(a[i] > T.float32(0.0), FLAG),
                        T.float32(1.0),
                        T.float32(0.0),
                    )
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_mask_with_scalar_bool_equality_uses_the_truth_table():
    """A mask-vs-scalar EQ cannot be a vcmp: the VMI element set has no bool."""
    stages, error = _lower_capture(_k_mask_with_scalar_eq())
    assert error is None, f"a mask/scalar EQ must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vand", 0) >= 1, f"the truth table needs vand: {names}"
    assert names.get("tl.vmi.vsel", 0) >= 1, f"the Select must use vsel: {names}"


def _k_serial_outside_unit(E=256, lanes=64, iters=8):
    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for _j in T.serial(iters):
                    for i in T.Parallel(E):
                        y[i] = a[i] + T.float32(1.0)
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_outer_serial_range_makes_a_unit_predicate_provable():
    """`j < iters` rests on the enclosing serial range, not on the unit.

    Verify and Legalize must carry the same lexical facts, or a predicate the
    normaliser would clear is rejected before it ever gets there.
    """
    state = {}

    def pre(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.SERIAL and "serial_var" not in state:
            state["serial_var"] = node.loop_var
        return None

    def post(node):
        if isinstance(node, tvm.tirx.BufferStore) and "serial_var" in state:
            return tvm.tirx.BufferStore(
                node.buffer,
                node.value,
                node.indices,
                state["serial_var"] < tvm.tirx.IntImm("int32", 8),
            )
        return None

    func = _k_serial_outside_unit()
    body = tvm.tirx.stmt_functor.ir_transform(func.body, pre, post, ["tirx.For", "tirx.BufferStore"])
    stages, error = _lower_capture(func.with_body(body))
    assert error is None, f"an outer-range predicate must be admitted, got: {error}"
    predicated = [
        node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferStore)) if node.predicate is not None
    ]
    assert not predicated, "Legalize must clear the outer-range predicate"


def _append_region_level_access(func, kind, make_predicate):
    """Append one region-level access (a load or a store) inside SIMD_VF.

    A region-level access sits beside the Parallel unit, so its predicate goes
    through the same admission as a unit access -- but not through the unit
    visitor, which is why it used to escape the reference check.
    """
    state = {}

    def post(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope() in ("shared", "shared.dyn"):
            state.setdefault("store_buffer", node.buffer)
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.scope() in ("shared", "shared.dyn"):
            state.setdefault("load_buffer", node.buffer)
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            load_buf = state["load_buffer"]
            store_buf = state["store_buffer"]
            predicate = make_predicate()
            if kind == "load":
                stmt = tvm.tirx.Evaluate(tvm.tirx.BufferLoad(load_buf, [tvm.tirx.IntImm("int32", 0)], predicate=predicate))
            else:
                value = tvm.tirx.BufferLoad(load_buf, [tvm.tirx.IntImm("int32", 0)])
                stmt = tvm.tirx.BufferStore(
                    store_buf,
                    value,
                    [tvm.tirx.IntImm("int32", 0)],
                    predicate=predicate,
                )
            body = node.body
            new_body = tvm.tirx.SeqStmt(list(body.seq) + [stmt]) if isinstance(body, tvm.tirx.SeqStmt) else tvm.tirx.SeqStmt([body, stmt])
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                new_body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=node.annotations,
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.BufferStore", "tirx.BufferLoad", "tirx.SBlock"])
    return func.with_body(body)


@pytest.mark.pto
def test_region_level_load_predicate_with_undefined_variable_is_rejected():
    """The predicate reference check must run before the truth proof.

    `True | ghost` folds to True, so a proof-first order (or a walk that only
    covers stores) accepts the load and Legalize then deletes the field together
    with the undefined variable.
    """
    ghost = tvm.tirx.Var("ghost_flag", "bool")
    func = _append_region_level_access(
        _k_plain_two_stores(),
        "load",
        lambda: tvm.tirx.Or(tvm.tirx.const(True, "bool"), ghost),
    )
    _, error = _lower_capture(func, expect_error=True)
    assert error is not None, "an undefined variable in a load predicate must be rejected"
    assert "not defined in the lexical scope" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_region_level_false_load_predicate_is_rejected():
    func = _append_region_level_access(_k_plain_two_stores(), "load", lambda: tvm.tirx.const(False, "bool"))
    _, error = _lower_capture(func, expect_error=True)
    assert error is not None, "a false load predicate must be rejected"
    assert "not provably true" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_region_level_true_load_predicate_is_admitted_and_cleared():
    func = _append_region_level_access(_k_plain_two_stores(), "load", lambda: tvm.tirx.const(True, "bool"))
    stages, error = _lower_capture(func)
    assert error is None, f"a provably-true load predicate must be admitted, got: {error}"
    predicated = [node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferLoad)) if node.predicate is not None]
    assert not predicated, "Legalize must clear the admitted load predicate"


def _k_region_level_bind(E=256, lanes=64):
    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                # The Bind is the point of this kernel: the predicate that
                # consumes it is injected below.
                flag = T.float32(1.0) > T.float32(0.0)  # noqa: F841
                for i in T.Parallel(E):
                    y[i] = a[i] + T.float32(1.0)
                y[0] = a[0] + T.float32(2.0)
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_region_level_bind_makes_a_predicate_provable():
    """A region-level scalar Bind is a fact for the predicate walk."""
    state = {}

    def post(node):
        if isinstance(node, tvm.tirx.Bind) and node.value.dtype == tvm.DataType("bool"):
            state.setdefault("flag", node.var)
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope() in ("shared", "shared.dyn") and "flag" in state and node.indices:
            return None
        return None

    def post_store(node):
        if (
            isinstance(node, tvm.tirx.BufferStore)
            and node.buffer.scope() in ("shared", "shared.dyn")
            and "flag" in state
            and isinstance(node.indices[0], tvm.tirx.IntImm)
        ):
            return tvm.tirx.BufferStore(node.buffer, node.value, node.indices, predicate=state["flag"])
        return None

    func = _k_region_level_bind()
    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.Bind"])
    func = func.with_body(body)
    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post_store, ["tirx.BufferStore"])
    stages, error = _lower_capture(func.with_body(body))
    assert error is None, f"a Bind-backed predicate must be admitted, got: {error}"
    predicated = [
        node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferStore)) if node.predicate is not None
    ]
    assert not predicated, "Legalize must clear a Bind-backed predicate"


# ---------------------------------------------------------------------------
# Parameter alias contract: two parameters are independent only under an
# explicit no-alias promise, and a non-restrict parameter cancels it.
# ---------------------------------------------------------------------------


def _k_two_shared_params(E=256, lanes=64):
    """Condition reads one UB buffer, two stores write another."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[i] = T.float32(0.0)
                        y[i] = T.float32(1.0)
            T.copy(y, Y)

    return main


def _as_shared_params(func, names=("a", "y"), extent=256):
    """Turn two UB allocations into shared-scope parameters.

    The frontend cannot declare a shared-scope parameter, so the two buffers a
    conditional reads and writes are hoisted into the ABI: now the caller may
    pass the same storage twice, and the pass must not assume independence.
    """
    f32 = tvm.DataType("float32")
    old = {}
    new = {}
    handles = {}

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.name in names:
                    old[buffer.name] = buffer

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    for name in names:
        if name not in old:
            raise AssertionError(f"buffer `{name}` not found among the allocations")
        var = tvm.tirx.Var(
            name + "_param",
            tvm.ir.PointerType(tvm.ir.PrimType(f32), "shared.dyn"),
        )
        handles[name] = var
        # A shared-scope parameter has no allocation base to infer from, so the
        # ABI must state the alignment the masked vstore needs.
        new[name] = tvm.tirx.decl_buffer((extent,), "float32", data=var, scope="shared.dyn", data_alignment=64)

    def post(node):
        if isinstance(node, tvm.tirx.SBlock):
            kept = [b for b in node.alloc_buffers if b.name not in names]
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                init=node.init,
                alloc_buffers=kept,
                match_buffers=node.match_buffers,
                annotations=node.annotations,
            )
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.name in names:
            return tvm.tirx.BufferStore(new[node.buffer.name], node.value, node.indices, node.predicate)
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name in names:
            return tvm.tirx.BufferLoad(new[node.buffer.name], node.indices, node.predicate)
        return None

    body = tvm.tirx.stmt_functor.ir_transform(
        func.body,
        None,
        post,
        ["tirx.SBlock", "tirx.BufferStore", "tirx.BufferLoad"],
    )
    buffer_map = {k: v for k, v in func.buffer_map.items()}
    params = list(func.params)
    for name in names:
        params.append(handles[name])
        buffer_map[handles[name]] = new[name]
    return tvm.tirx.PrimFunc(params, body, func.ret_type, buffer_map, func.attrs)


def _assert_noalias(func):
    return func.with_attr("tirx.noalias", tvm.tirx.IntImm("int32", 1))


def _assert_non_restrict(func, name="a_param"):
    var = next(v for v in func.params if v.name == name)
    return func.with_attr("tl.non_restrict_params", [var])


def _snapshot_count(func):
    """Count entry snapshots in the Legalize output of a shared-parameter kernel.

    A shared-scope parameter has no UB allocation root, so the later codegen
    legality pass cannot prove its masked-store alignment and the full pipeline
    stops there. The alias contract itself is decided by Legalize -- its conflict
    test drives the snapshot -- and that stage is captured before the failure, so
    the capture is read with `expect_error=True` and only the Legalize module is
    inspected.
    """
    stages, error = _lower_capture(func, expect_error=True)
    legalized = stages.get("legalize")
    assert legalized is not None, f"Legalize did not run: {error}"
    return len(_condition_snapshots(legalized))


@pytest.mark.pto
def test_distinct_params_without_a_noalias_promise_keep_the_snapshot():
    """No `tirx.noalias` at this stage: the caller may pass one storage twice."""
    assert _snapshot_count(_as_shared_params(_k_two_shared_params())) == 1


@pytest.mark.pto
def test_distinct_params_under_a_noalias_promise_skip_the_snapshot():
    """An explicit no-alias promise makes two parameters independent."""
    func = _assert_noalias(_as_shared_params(_k_two_shared_params()))
    assert _snapshot_count(func) == 0


@pytest.mark.pto
def test_non_restrict_param_cancels_the_noalias_promise():
    """`tl.non_restrict_params` says the pointer may alias after all."""
    func = _assert_non_restrict(_assert_noalias(_as_shared_params(_k_two_shared_params())))
    assert _snapshot_count(func) == 1


def _k_region_store_in_serial(E=256, lanes=64, iters=8):
    """A store beside the unit, inside the enclosing serial loop."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for _j in T.serial(iters):
                    for i in T.Parallel(E):
                        y[i] = a[i] + T.float32(1.0)
                    a[0] = a[0] + T.float32(2.0)
            T.copy(y, Y)

    return main


def _with_region_store_predicate(func, make_predicate):
    """Attach `make_predicate(j)` to the region-level store in the serial body."""

    state = {}

    def pre(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.SERIAL and "serial_var" not in state:
            state["serial_var"] = node.loop_var
        return None

    def post(node):
        if (
            isinstance(node, tvm.tirx.BufferStore)
            and node.buffer.scope() in ("shared", "shared.dyn")
            and "serial_var" in state
            and len(node.indices) == 1
            and isinstance(node.indices[0], tvm.tirx.IntImm)
        ):
            return tvm.tirx.BufferStore(
                node.buffer,
                node.value,
                node.indices,
                predicate=make_predicate(state["serial_var"]),
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, pre, post, ["tirx.For", "tirx.BufferStore"])
    return func.with_body(body)


@pytest.mark.pto
def test_region_level_store_predicate_over_a_serial_range_is_admitted():
    """`j < 8` on a store beside the unit, provable from the enclosing shell."""
    func = _with_region_store_predicate(
        _k_region_store_in_serial(),
        lambda j: j < tvm.tirx.IntImm("int32", 8),
    )
    stages, error = _lower_capture(func)
    assert error is None, f"a range-provable store predicate must be admitted, got: {error}"
    predicated = [
        node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferStore)) if node.predicate is not None
    ]
    assert not predicated, "Legalize must clear the admitted store predicate"


@pytest.mark.pto
def test_region_level_store_predicate_with_undefined_variable_is_rejected():
    """The store path and the load path share one reference check."""
    func = _with_region_store_predicate(
        _k_region_store_in_serial(),
        lambda j: tvm.tirx.Or(
            j < tvm.tirx.IntImm("int32", 8),
            tvm.tirx.Var("ghost_flag", "bool"),
        ),
    )
    _, error = _lower_capture(func, expect_error=True)
    assert error is not None, "an undefined variable in a store predicate must be rejected"
    assert "not defined in the lexical scope" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_predicate_with_an_unadmitted_node_is_rejected():
    """A predicate is an expression position: `True | ...` must not launder it.

    The oracle folds `True | (P % 2 == 0)` to True, so a truth-check-only
    admission would drop the whole predicate -- and with it an expression the
    vectorizer has no lowering for.
    """
    func = _k_plain_two_stores()
    param = func.params[0]
    func = _append_region_level_access(
        func,
        "load",
        lambda: tvm.tirx.Or(
            tvm.tirx.const(True, "bool"),
            tvm.tirx.EQ(tvm.tirx.floormod(param, 2), tvm.tirx.IntImm("int32", 0)),
        ),
    )
    _, error = _lower_capture(func, expect_error=True)
    assert error is not None, "an unadmitted node in a predicate must be rejected"
    assert "FloorMod" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_predicate_with_an_external_call_is_rejected():
    """An external call in a predicate could carry a side effect.

    Dropping a provably-true predicate would silently delete that call, so the
    whitelist must reject it before the truth proof.
    """
    func = _append_region_level_access(
        _k_plain_two_stores(),
        "load",
        lambda: tvm.tirx.Or(
            tvm.tirx.const(True, "bool"),
            tvm.tirx.call_extern("bool", "predicate_effect"),
        ),
    )
    _, error = _lower_capture(func, expect_error=True)
    assert error is not None, "an external call in a predicate must be rejected"
    assert "unsupported Call" in str(error), f"unexpected diagnostic: {error}"


# ---------------------------------------------------------------------------
# A Bind is an SSA snapshot: a bind over memory must stay opaque in a proof
# ---------------------------------------------------------------------------


def _ub_buffers(func):
    """(load buffer, store buffer) of the first UB access pair in the body."""
    state = {}

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.scope() in ("shared", "shared.dyn") and "load" not in state:
            state["load"] = node.buffer
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope() in ("shared", "shared.dyn") and "store" not in state:
            state["store"] = node.buffer

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return state.get("load"), state.get("store")


def _prepend_to_vf(func, make_stmts):
    """Prepend statements to the SIMD_VF body (region-level position)."""

    def post(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            extra = make_stmts()
            body = node.body
            new_body = tvm.tirx.SeqStmt(extra + list(body.seq)) if isinstance(body, tvm.tirx.SeqStmt) else tvm.tirx.SeqStmt(extra + [body])
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                new_body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=node.annotations,
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.SBlock"])
    return func.with_body(body)


def _snapshot_predicate_scene(func, snapshot_value, predicate_from_snapshot):
    """`old = <read>; buf[0] = 0; Evaluate(load(predicate = f(old)))`."""
    load_buf, store_buf = _ub_buffers(func)
    if store_buf is None:
        store_buf = load_buf
    index = tvm.tirx.IntImm("int32", 0)

    def make_stmts():
        old = tvm.tirx.Var("old_snapshot", snapshot_value.dtype)
        return [
            tvm.tirx.Bind(old, snapshot_value),
            tvm.tirx.BufferStore(store_buf, tvm.tirx.const(0.0, "float32"), [index]),
            tvm.tirx.Evaluate(
                tvm.tirx.BufferLoad(
                    load_buf,
                    [index],
                    predicate=predicate_from_snapshot(old),
                )
            ),
        ]

    return _prepend_to_vf(func, make_stmts)


@pytest.mark.pto
def test_value_snapshot_before_a_write_is_not_provable():
    """`old == a[0]` reads memory again: the snapshot must not be re-expanded."""
    func = _k_plain_two_stores()
    load_buf, _ = _ub_buffers(func)
    index = tvm.tirx.IntImm("int32", 0)
    scene = _snapshot_predicate_scene(
        func,
        tvm.tirx.BufferLoad(load_buf, [index]),
        lambda old: tvm.tirx.EQ(old, tvm.tirx.BufferLoad(load_buf, [index])),
    )
    _, error = _lower_capture(scene, expect_error=True)
    assert error is not None, "a comparison against a value snapshot must not be proven from a re-read"
    assert "not provably true" in str(error), f"unexpected diagnostic: {error}"


@pytest.mark.pto
def test_bool_snapshot_before_a_write_is_not_provable():
    """The bool form: `old_flag = a[0] > 0` must stay an opaque snapshot too."""
    func = _k_plain_two_stores()
    load_buf, _ = _ub_buffers(func)
    index = tvm.tirx.IntImm("int32", 0)
    scene = _snapshot_predicate_scene(
        func,
        tvm.tirx.GT(tvm.tirx.BufferLoad(load_buf, [index]), tvm.tirx.const(0.0, "float32")),
        lambda old: old,
    )
    _, error = _lower_capture(scene, expect_error=True)
    assert error is not None, "a bool snapshot must not be re-read as its source"
    assert "not provably true" in str(error), f"unexpected diagnostic: {error}"


# ---------------------------------------------------------------------------
# Fact scoping: a branch-local Bind must not survive its branch
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_branch_local_bind_does_not_escape_its_scope():
    """`if ...: proof = True` then a predicate outside the branch uses `proof`."""
    func = _k_plain_two_stores()
    load_buf, _ = _ub_buffers(func)
    index = tvm.tirx.IntImm("int32", 0)
    # A constant condition would be folded away before Verify; use the kernel
    # parameter so the branch really exists at the admission walk.
    param = func.params[0]
    state = {}

    def make_stmts():
        proof = tvm.tirx.Var("branch_proof", "bool")
        state["proof"] = proof
        branch = tvm.tirx.IfThenElse(
            tvm.tirx.GT(param, tvm.tirx.IntImm("int32", 0)),
            tvm.tirx.Bind(proof, tvm.tirx.const(True, "bool")),
            None,
        )
        return [
            branch,
            tvm.tirx.Evaluate(tvm.tirx.BufferLoad(load_buf, [index], predicate=proof)),
        ]

    scene = _prepend_to_vf(func, make_stmts)
    _, error = _lower_capture(scene, expect_error=True)
    assert error is not None, "a branch-local Bind must not be visible after the branch"
    assert "not defined in the lexical scope" in str(error), f"unexpected diagnostic: {error}"


def _k_bind_before_vf(E=256, lanes=64):
    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[i] = a[i] + T.float32(1.0)
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_bind_defined_before_the_region_stays_a_fact_inside_it():
    """A Bind/loop range valid around the VF must be usable inside it."""
    func = _k_bind_before_vf()
    load_buf, _ = _ub_buffers(func)
    proof = tvm.tirx.Var("outer_proof", "bool")

    # Insert `proof = True` as a sibling *before* the SIMD_VF block.
    def prepend_before_vf(node):
        if isinstance(node, tvm.tirx.SeqStmt) and any(
            isinstance(child, tvm.tirx.SBlock) and child.name_hint == "SIMD_VF" for child in node.seq
        ):
            return tvm.tirx.SeqStmt([tvm.tirx.Bind(proof, tvm.tirx.const(True, "bool"))] + list(node.seq))
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, prepend_before_vf, ["tirx.SeqStmt"])
    func = func.with_body(body)

    scene = _append_region_level_access(
        func,
        "load",
        lambda: proof,
    )
    stages, error = _lower_capture(scene)
    assert error is None, f"an outer-VF Bind must remain a fact, got: {error}"
    predicated = [node for node in _collect(stages["legalize"], lambda n: isinstance(n, tvm.tirx.BufferLoad)) if node.predicate is not None]
    assert not predicated, "Legalize must clear the outer-VF-backed predicate"


def _k_runtime_offset_destination(E=256, lanes=64, width=512):
    """A conditional destination shifted by a *runtime* uniform offset."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        K: T.int32,
        Y: T.Tensor((width,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            y = T.alloc_shared((width,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[i + 8 * K] = a[i]
            T.copy(y, Y)

    return main


@pytest.mark.pto
def test_runtime_uniform_offset_destination_folds():
    """A uniform-but-runtime destination offset is still per-lane and pairable.

    `i + 8 * K` keeps the lane coefficient at 1, so the access is continuous and
    the read-back stays per-lane; only the offset is unknown at compile time. The
    admission is the continuity proof, not a constant-offset form rule.
    """
    stages, error = _lower_capture(_k_runtime_offset_destination())
    assert error is None, f"a runtime uniform offset must lower, got: {error}"

    legalized = stages["legalize"]
    assert legalized is not None
    branches = [node for node in _collect(legalized, lambda n: isinstance(n, tvm.tirx.IfThenElse))]
    assert not branches, "the conditional must be folded into a value choice"
    assert "vsel" in str(stages["vectorize"]), "the folded store must lower to a lane-wise select"
