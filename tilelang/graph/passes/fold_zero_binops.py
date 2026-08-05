"""Fold ``add(0, x) → x`` and ``add(x, 0) → x`` across Relax functions.

Matches both ``relax.add`` (pre-LegalizeOps) and ``call_tir(add_fn, [a, b])``
(post-LegalizeOps) so the same pass works either side of legalization.
"""

import logging

import numpy as np

from tilelang import tvm as tvm
from tvm import relax, tirx as tir

logger = logging.getLogger(__name__)


def _is_const_zero(expr) -> bool:
    """True iff ``expr`` is a Relax Constant tensor whose values are all 0."""
    if not isinstance(expr, relax.Constant):
        return False
    try:
        arr = expr.data.numpy()
    except Exception:
        return False
    return bool(np.all(arr == 0))


def _resolve_to_constant(expr, var_to_value: dict) -> "relax.Constant | None":
    """Walk var bindings until we either find a Constant or give up."""
    seen = set()
    while isinstance(expr, relax.Var):
        if expr in seen:
            return None
        seen.add(expr)
        v = var_to_value.get(expr)
        if v is None:
            return None
        expr = v
    if isinstance(expr, relax.Constant):
        return expr
    return None


def _is_pure_add_primfunc(fn, gv_name: str) -> bool:
    """Conservative check: is ``fn`` exactly the topi-emitted add PrimFunc?

    We require the global var name to be exactly 'add' (LegalizeOps gives
    standalone Relax adds this name, and FuseOps doesn't fuse them any
    further if their inputs are external).  The body must have 3 buffer
    params matching the cache_position add pattern.
    """
    if not isinstance(fn, tir.PrimFunc):
        return False
    if gv_name != "add":
        return False
    return len(fn.params) == 3


@relax.expr_functor.mutator
class _ZeroBinopFolder(relax.PyExprMutator):
    def __init__(self, mod, var_to_value: dict):
        super().__init__(mod)
        self.mod = mod
        self.var_to_value = var_to_value
        self.folded_count = 0

    def _fold(self, a, b):
        ac = _resolve_to_constant(a, self.var_to_value)
        if ac is not None and _is_const_zero(ac):
            self.folded_count += 1
            return b
        bc = _resolve_to_constant(b, self.var_to_value)
        if bc is not None and _is_const_zero(bc):
            self.folded_count += 1
            return a
        return None

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if not isinstance(call, relax.Call):
            return call
        op_name = call.op.name if hasattr(call.op, "name") else None

        if op_name == "relax.add" and len(call.args) == 2:
            folded = self._fold(call.args[0], call.args[1])
            if folded is not None:
                return folded

        if op_name in ("relax.call_tir", "relax.call_tir_inplace") and len(call.args) >= 2:
            gv = call.args[0]
            arg_tup = call.args[1]
            if (
                isinstance(gv, relax.GlobalVar)
                and isinstance(arg_tup, relax.Tuple)
                and len(arg_tup.fields) == 2
                and _is_pure_add_primfunc(self.mod.functions.get(gv), gv.name_hint)
            ):
                folded = self._fold(arg_tup.fields[0], arg_tup.fields[1])
                if folded is not None:
                    return folded
        return call


def fold_zero_binops(mod):
    """Module pass: fold ``add(0, x) → x`` across all Relax functions.

    Builds a var→value map per function so the folder can resolve vars that
    ultimately bind to a Constant (Relax SSA usually binds constants to vars
    before using them).
    """
    total_folded = 0
    for gv, fn in mod.functions.items():
        if not isinstance(fn, relax.Function):
            continue
        var_to_value: dict = {}
        if isinstance(fn.body, relax.SeqExpr):
            for block in fn.body.blocks:
                for b in block.bindings:
                    if isinstance(b, relax.VarBinding):
                        var_to_value[b.var] = b.value
        folder = _ZeroBinopFolder(mod, var_to_value)
        mod[gv] = folder.visit_expr(fn)
        total_folded += folder.folded_count
    logger.debug("fold_zero_binops folded %d ops", total_folded)
    return relax.transform.DeadCodeElimination()(mod)
