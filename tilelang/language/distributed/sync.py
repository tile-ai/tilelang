from __future__ import annotations

from enum import Enum

from tvm import tir
from tvm.tir import PrimExpr, address_of


class BinaryRelation(Enum):
    EQ = 0
    NE = 1
    GE = 2
    LE = 3
    GT = 4
    LT = 5


class WaitScope(Enum):
    SYS = 0
    GPU = 1


class WaitSemantics(Enum):
    ACQUIRE = 0
    VOLATILE = 1


def _enum_value(enum_cls, value, name: str) -> int:
    if isinstance(value, enum_cls):
        return value.value
    if isinstance(value, str):
        key = value.upper()
        if key in enum_cls.__members__:
            return enum_cls[key].value
    raise ValueError(f"Unsupported {name}: {value}")


def _wait(
    relation: BinaryRelation,
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.wait"),
        relation.value,
        address_of(value),
        expected,
        peer,
        _enum_value(WaitScope, scope, "wait scope"),
        _enum_value(WaitSemantics, semantics, "wait semantics"),
    )


def wait_eq(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value == expected."""
    return _wait(BinaryRelation.EQ, value, expected, peer, scope=scope, semantics=semantics)


def wait_ne(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value != expected."""
    return _wait(BinaryRelation.NE, value, expected, peer, scope=scope, semantics=semantics)


def wait_ge(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value >= expected."""
    return _wait(BinaryRelation.GE, value, expected, peer, scope=scope, semantics=semantics)


def wait_le(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value <= expected."""
    return _wait(BinaryRelation.LE, value, expected, peer, scope=scope, semantics=semantics)


def wait_gt(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value > expected."""
    return _wait(BinaryRelation.GT, value, expected, peer, scope=scope, semantics=semantics)


def wait_lt(
    value: PrimExpr,
    expected: PrimExpr,
    peer: PrimExpr | None = -1,
    *,
    scope: WaitScope | str = WaitScope.SYS,
    semantics: WaitSemantics | str = WaitSemantics.ACQUIRE,
):
    """Wait until value < expected."""
    return _wait(BinaryRelation.LT, value, expected, peer, scope=scope, semantics=semantics)


def init_barrier_gpu(barrier: PrimExpr, expected: int):
    """Initialize a barrier for GPU-level synchronization."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.init_barrier_gpu"), address_of(barrier), expected)


def arrive_barrier_gpu(barrier: PrimExpr):
    """Arrive at a barrier for GPU-level synchronization."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.arrive_barrier_gpu"), address_of(barrier))


def wait_barrier_gpu(barrier: PrimExpr):
    """Wait at a barrier for GPU-level synchronization."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.wait_barrier_gpu"), address_of(barrier))


def sync_barrier_gpu(barrier: PrimExpr):
    """Synchronize at a GPU barrier (arrive + wait)."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.sync_barrier_gpu"), address_of(barrier))


def barrier_blocks(barrier: PrimExpr):
    """Barrier all blocks at a system-level barrier with fence."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.barrier_blocks"), address_of(barrier), 1)


def sync_blocks(barrier: PrimExpr):
    """Synchronize all blocks at a system-level barrier without fence."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.barrier_blocks"), address_of(barrier), 0)
