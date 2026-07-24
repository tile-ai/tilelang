from tvm import tirx


__all__ = [
    "pdl_trigger",
    "pdl_sync",
]


def pdl_trigger() -> tirx.PrimExpr:
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.pdl_trigger"),
    )


def pdl_sync() -> tirx.PrimExpr:
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.pdl_sync"),
    )
