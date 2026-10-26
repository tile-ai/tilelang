"""TileLang query adapters over the unmodified TIRx symbolic analyzer."""

import tvm_ffi
from tvm_tirx import sym as _native
from tvm_tirx.sym import *
from tvm_tirx.sym import _ffi_api


def _query(name, *args):
    return tvm_ffi.get_global_func(f"tl.sym.{name}")(*args)


class Analyzer(_native.Analyzer):
    def simplify(self, expr, steps=2):
        return _query("Simplify", self, expr, steps)

    def rewrite_simplify(self, expr):
        return _query("RewriteSimplify", self, expr)

    def canonical_simplify(self, expr):
        return _query("CanonicalSimplify", self, expr)

    def const_int_bound(self, expr):
        return _query("ConstIntBound", self, expr)

    def modular_set(self, expr):
        return _query("ModularSet", self, expr)

    def int_set(self, expr, dom_map=None):
        return _query("IntSet", self, expr, dom_map)

    def can_prove(self, expr, strength=_native.ProofStrength.DEFAULT):
        return _query("CanProve", self, expr, strength)

    def can_prove_equal(self, lhs, rhs):
        return _query("CanProveEqual", self, lhs, rhs)

    def bind(self, var, expr, allow_override=False):
        _query("Bind", self, var, expr, allow_override)

    def constraint_scope(self, constraint):
        return super().constraint_scope(_query("NormalizeConstraint", self, constraint))

    def get_smtlib2(self, expr=None):
        if expr is not None:
            expr = _query("NormalizeConstraint", self, expr)
        return super().get_smtlib2(expr)

    def clone(self):
        result = Analyzer.__new__(Analyzer)
        result.__init_handle_by_constructor__(_ffi_api.AnalyzerClone, self)
        return result


def __getattr__(name):
    return getattr(_native, name)
