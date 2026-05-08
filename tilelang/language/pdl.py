from tvm import tir


__all__ = [
    "pdl_trigger",
    "pdl_sync",
]


def pdl_trigger() -> tir.PrimExpr:
    return tir.call_intrin(
        "void",
        tir.op.Op.get("tl.pdl_trigger"),
    )


def pdl_sync() -> tir.PrimExpr:
    return tir.call_intrin(
        "void",
        tir.op.Op.get("tl.pdl_sync"),
    )
