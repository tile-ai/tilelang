"""Multimem operations (NVSwitch SHARP multicast) using layout-aware lowering.

These operations use T.copy's ParallelOp + InferLayout + VectorizeLoop pipeline
to correctly handle fragment layouts, then post-process to emit multimem instructions.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
from tvm import tir
from tvm.tir import PrimExpr, address_of
from tilelang.utils.language import to_buffer_region


class MultimemReduceOp(Enum):
    ADD = 0
    MIN = 1
    MAX = 2
    NONE = -1  # plain store (no reduction), for multimem_tma_store


class _MultimemMode(Enum):
    LD_REDUCE = 0
    ST = 1
    RED = 2
    TMA_STORE = 3
    TMA_RED_STORE = 4


def _multimem_impl(src, dst, mode: _MultimemMode, reduce_op: MultimemReduceOp = MultimemReduceOp.NONE):
    """Shared implementation for all multimem operations.

    Converts src/dst to buffer regions and emits the tl.tileop.multimem intrinsic.

    Args:
        src: Source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Destination (Buffer, BufferLoad with slice, or BufferRegion)
        mode: 0=kLdReduce, 1=kSt, 2=kRed
        reduce_op: 0=ADD, 1=MIN, 2=MAX
    """
    src_region = to_buffer_region(src, access_type="r")
    dst_region = to_buffer_region(dst, access_type="w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.multimem"),
        src_region,
        dst_region,
        mode.value,
        reduce_op.value,
    )


def multimem_ld_reduce(src, dst, reduce_op: MultimemReduceOp = MultimemReduceOp.ADD):
    """Load-reduce from multicast address into local buffer.

    Uses T.copy's layout inference to handle fragment layouts correctly.
    Each thread issues 128-bit multimem instructions after vectorization.

    Args:
        src: Multicast source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Local destination (Buffer, BufferLoad with slice, or BufferRegion)
        reduce_op: Reduction operation: 0=ADD, 1=MIN, 2=MAX.
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.LD_REDUCE, reduce_op=reduce_op)


def multimem_st(src, dst):
    """Store to multicast address (broadcast to all ranks).

    Args:
        src: Local source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Multicast destination (Buffer, BufferLoad with slice, or BufferRegion)
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.ST)


def multimem_red(src, dst, reduce_op: MultimemReduceOp = MultimemReduceOp.ADD):
    """Reduce into multicast address (accumulate without read-back).

    Args:
        src: Local source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Multicast destination (Buffer, BufferLoad with slice, or BufferRegion)
        reduce_op: Reduction operation: 0=ADD, 1=MIN, 2=MAX.
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.RED, reduce_op=reduce_op)


def multimem_tma_store(src, dst, reduce_op: MultimemReduceOp | None = None):
    """Async bulk TMA store from shared memory to multicast global address.

    CTA-collective: a single thread emits one PTX instruction per call.
    Uses bulk_group completion (fence.proxy.async + commit_group + wait).

    Args:
        src: Shared memory source (Buffer, BufferLoad or BufferRegion, shared scope)
        dst: Multicast global destination (Buffer, BufferLoad or BufferRegion, global scope)
        reduce_op: None for plain store (broadcast), MultimemReduceOp.ADD/MIN/MAX for reduce-accumulate

    NOTE: This instruction requires Hopper+ and CUDA toolkit 13.x.
    (For unsatisfied CTK version, a hack is to use plain TMA store to mcast vaddr.)
    """
    if reduce_op is None:
        return _multimem_impl(src, dst, mode=_MultimemMode.TMA_STORE)
    return _multimem_impl(src, dst, mode=_MultimemMode.TMA_RED_STORE, reduce_op=reduce_op)


def multimem_signal(addr, value: PrimExpr, dtype_tag: Literal["uint32", "uint64"] = "uint32"):
    return tir.call_extern("handle", f"tl::multimem::Signal<{dtype_tag}>::run", address_of(addr), value)
