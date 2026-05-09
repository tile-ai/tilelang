from __future__ import annotations

from tvm import tir
from tvm.tir import PrimExpr, IntImm, address_of


def put_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    dst_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Put to a remote buffer with unrolled loop."""
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.put"),
        src,
        dst,
        size,
        dst_pe,
        unroll_factor,
        "warp",
        enable_aggressive_vectorize,
    )


def get_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    src_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Get from a remote buffer with unrolled loop."""
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.get"),
        src,
        dst,
        size,
        src_pe,
        unroll_factor,
        "warp",
        enable_aggressive_vectorize,
    )


def put_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, dst_pe: PrimExpr | IntImm | None = -1):
    """Put to a remote buffer."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.put"), src, dst, size, dst_pe, 0, "block", True)


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | IntImm | None = -1):
    """Get from a remote buffer."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.get"), src, dst, size, src_pe, 0, "block", True)


def ld(
    src: PrimExpr,
    value: PrimExpr,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    nc: bool = False,
    src_pe: tir.PrimExpr | tir.IntImm | None = -1,
):
    """Load a value from an address with explicit PTX scope and semantic."""
    assert scope in ["cta", "gpu", "sys"], "Scope must be one of 'cta', 'gpu', or 'sys'."
    assert sem in ["weak", "volatile", "acquire", "relaxed"], (
        "Semantic must be one of 'weak', 'volatile', 'acquire', or 'relaxed'."
    )
    scope_id = {"cta": 0, "gpu": 1, "sys": 2}[scope]
    sem_id = {"weak": 0, "volatile": 1, "acquire": 2, "release": 3, "relaxed": 4}[sem]
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.ld"), address_of(src), value, sem_id, scope_id, int(na), int(nc), src_pe
    )


def st(
    dst: PrimExpr,
    value: PrimExpr,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    dst_pe: tir.PrimExpr | tir.IntImm | None = -1,
):
    """Store a value to an address with explicit PTX scope and semantic."""
    assert scope in ["cta", "gpu", "sys"], "Scope must be one of 'cta', 'gpu', or 'sys'."
    assert sem in ["weak", "volatile", "release", "relaxed"], (
        "Semantic must be one of 'weak', 'volatile', 'release', or 'relaxed'."
    )
    scope_id = {"cta": 0, "gpu": 1, "sys": 2}[scope]
    sem_id = {"weak": 0, "volatile": 1, "acquire": 2, "release": 3, "relaxed": 4}[sem]
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.st"), address_of(dst), value, sem_id, scope_id, int(na), dst_pe)


def atom_add(target: PrimExpr, value: PrimExpr, scope: str = "gpu", sem: str = "relaxed"):
    """Perform a scoped uint32 atomic add and return the previous value."""
    assert scope in ["gpu", "sys"], "Scope must be one of 'gpu', or 'sys'."
    assert sem in ["relaxed", "acquire", "release", "acq_rel"], (
        "Semantic must be one of 'relaxed', 'acquire', 'release', or 'acq_rel'."
    )
    return tir.call_intrin("uint32", tir.op.Op.get("tl.atom_add"), address_of(target), value, sem, scope)
