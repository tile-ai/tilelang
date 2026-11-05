"""Translate Carver's inclusive integer bounds to native analysis premises."""

import tvm_ffi
from tvm_tirx import ir, sym


def const_bound_context(bounds):
    """Build one context from the final bounds selected for each Var identity.

    ConstIntBound uses sentinels for infinity. Neither sentinel is a typed
    integer constant to materialize in the IR. Finite bounds outside the
    variable's dtype are represented by tautological or impossible comparisons
    against its representable endpoints, without a narrowing/wrapping IntImm.
    """
    context = sym.ConstraintContext()
    unsigned_code = tvm_ffi.dtype("uint8").type_code
    for variable, bound in bounds.items():
        if not isinstance(variable.ty, ir.PrimType):
            raise TypeError("Carver index bounds require a primitive variable")
        dtype = variable.ty.dtype
        if not dtype.is_integer or dtype.lanes != 1:
            raise TypeError("Carver index bounds require a scalar integer variable")
        unsigned = dtype.type_code == unsigned_code
        minimum = 0 if unsigned else -(1 << (dtype.bits - 1))
        maximum = (1 << (dtype.bits if unsigned else dtype.bits - 1)) - 1
        lower, upper = int(bound.min_value), int(bound.max_value)
        if lower != sym.ConstIntBound.NEG_INF:
            if lower > maximum:
                context.append(sym.PredicateConstraint(variable > ir.const(maximum, dtype)))
            elif lower > minimum:
                context.append(sym.PredicateConstraint(variable >= ir.const(lower, dtype)))
        if upper != sym.ConstIntBound.POS_INF:
            if upper < minimum:
                context.append(sym.PredicateConstraint(variable < ir.const(minimum, dtype)))
            elif upper < maximum:
                context.append(sym.PredicateConstraint(variable <= ir.const(upper, dtype)))
    return context
