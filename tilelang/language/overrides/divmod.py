"""Diagnose floating-point use of TIR's integer-only % and // operators."""

from functools import wraps

from tvm.tirx.expr import ExprOp

from tilelang.language.tir.op import _check_integer_divmod


def _checked_operator(original, symbol):
    @wraps(original)
    def checked(self, other):
        _check_integer_divmod(symbol, self, other)
        return original(self, other)

    return checked


# Both parsers and direct Python expression construction use ExprOp. In TVM,
# // uses generic dispatch but % calls FFI directly, so guard only these four
# methods rather than replacing generic dispatch or the shared FFI functions.
for _method, _symbol in [("__mod__", "%"), ("__rmod__", "%"), ("__floordiv__", "//"), ("__rfloordiv__", "//")]:
    setattr(ExprOp, _method, _checked_operator(getattr(ExprOp, _method), _symbol))
