from tilelang.language import _intrinsics
from tvm_tirx import ir as _ir


__all__ = [
    "pdl_trigger",
    "pdl_sync",
]


def pdl_trigger() -> _ir.Expr:
    return _intrinsics.call_intrin(
        "void",
        _ir.Op.get("tl.pdl_trigger"),
    )


def pdl_sync() -> _ir.Expr:
    return _intrinsics.call_intrin(
        "void",
        _ir.Op.get("tl.pdl_sync"),
    )
