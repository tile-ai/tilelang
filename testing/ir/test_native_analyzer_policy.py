"""TileLang's native proof policy and the pinned upstream's known limitations."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang._analyzer import create_analyzer


def test_tilelang_policy_is_local_and_clone_preserves_it():
    native = sym.Analyzer()
    native.set_z3_default_fallback(False)
    analyzer = create_analyzer()
    assert isinstance(analyzer, sym.Analyzer)
    # Consumers still receive the target's simplification rules.
    x = ir.Var("x", "int32")
    assert tvm_ffi.structural_equal(analyzer.simplify(x + 0), native.simplify(x + 0))
    if analyzer.is_z3_enabled:
        y = ir.Var("y", "int32")
        predicate = x * x + y * y >= 2 * (x * y)
        clone = analyzer.clone()
        assert not native.can_prove(predicate)
        assert analyzer.can_prove(predicate) and clone.can_prove(predicate)
        clone.set_z3_default_fallback(False)
        assert not clone.can_prove(predicate)
        assert analyzer.can_prove(predicate)


@pytest.mark.xfail(strict=True, reason="Inherited upstream narrowing-cast proof defect; deferred from this rebase")
def test_tilelang_fallback_proves_only_with_sound_typed_constraints():
    analyzer = create_analyzer()
    if not analyzer.is_z3_enabled:
        pytest.skip("This build has no Z3 backend")
    x = ir.Var("x", "uint32")
    predicate = tirx.Cast("uint32", tirx.Cast("uint8", x)) == x
    assert not analyzer.can_prove(predicate)
    constraints = sym.ConstraintContext()
    constraints.append(sym.PredicateConstraint(x < ir.IntImm("uint32", 256)))
    analyzer.sync_constraint_context(constraints)
    assert analyzer.can_prove(predicate)
    analyzer.sync_constraint_context(sym.ConstraintContext())
    assert not analyzer.can_prove(predicate)


def test_tilelang_fallback_keeps_budget_and_native_fast_path():
    analyzer = create_analyzer()
    if not analyzer.is_z3_enabled:
        pytest.skip("This build has no Z3 backend")
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    # This is the native Analyzer's mathematical signed index domain.
    predicate = x * x + y * y >= 2 * (x * y)
    native_only = sym.Analyzer()
    native_only.set_z3_default_fallback(False)
    assert not native_only.can_prove(predicate)
    assert analyzer.can_prove(predicate)
    limited = create_analyzer()
    limited.set_z3_rlimit(1)
    assert not limited.can_prove(predicate)
    assert limited.can_prove(x == x)
