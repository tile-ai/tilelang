"""Reject unsafe multi-GEMM loop-indexed atomic partitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tilelang.tileir.errors import TileIRLoweringNotImplementedError
from tilelang.tileir.ir.ops import AtomicRMW, Copy, Gemm, Loop
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.passes.base import PassContext, walk_block

__all__ = ["dynamic_atomic_validation_pass"]


def _values(value: Any) -> Iterable[Value]:
    if isinstance(value, Value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _values(item)


def _depends_on(value: Any, target: Value, producers: dict[int, Any], seen: set[int]) -> bool:
    if not isinstance(value, Value):
        return False
    if value.id == target.id:
        return True
    if value.id in seen:
        return False
    seen.add(value.id)

    producer = producers.get(value.id)
    if producer is None:
        return False
    return any(_depends_on(operand, target, producers, seen) for field in producer.operands() for operand in _values(field))


def _collect_ops(block: Block, op_type: type) -> list[Any]:
    found: list[Any] = []

    def collect(op: Any) -> None:
        if isinstance(op, op_type):
            found.append(op)

    walk_block(block, collect)
    return found


def _atomic_value_source_ids(atomic: AtomicRMW, copies: list[Copy]) -> set[int]:
    source_ids = {id(atomic.val)}
    changed = True
    while changed:
        changed = False
        for copy in copies:
            if id(copy.dst) in source_ids and id(copy.src) not in source_ids:
                source_ids.add(id(copy.src))
                changed = True
    return source_ids


def dynamic_atomic_validation_pass(root: Block, ctx: PassContext) -> None:
    """Reject unsafe multi-GEMM atomics with serial-loop partitions.

    The current CUDA Tile IR toolchain can fail assembly or produce incorrect
    repeated-launch results when a loop writes an atomic value through multiple
    GEMM updates and changes the destination partition on every iteration.
    Refuse that combination before assembly instead of accepting an unsafe
    cubin. A single-GEMM atomic value remains supported.
    """

    producers: dict[int, Any] = {}

    def collect_producers(op: Any) -> None:
        for result in getattr(op, "results", ()):
            producers[result.id] = op

    walk_block(root, collect_producers)

    for loop in _collect_ops(root, Loop):
        if not loop.body.params:
            continue
        induction_var = loop.body.params[0]
        copies = _collect_ops(loop.body, Copy)
        gemms = _collect_ops(loop.body, Gemm)
        for atomic in _collect_ops(loop.body, AtomicRMW):
            source_ids = _atomic_value_source_ids(atomic, copies)
            gemm_updates = sum(id(gemm.acc) in source_ids for gemm in gemms)
            loop_indexed = any(_depends_on(index, induction_var, producers, set()) for index in atomic.dst_indices)
            if loop_indexed and gemm_updates > 1:
                raise TileIRLoweringNotImplementedError(
                    "TileIR backend does not yet support a loop-indexed atomic reduction "
                    "whose value receives multiple GEMM updates per iteration. Move the "
                    "reduction outside the loop or use another backend."
                )

    ctx.results["dynamic_atomic_validation"] = True
