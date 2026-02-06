"""The language interface for tl programs."""

from __future__ import annotations

from tvm import tir
from tvm.tir import address_of
from tvm.tir import PrimExpr, IntImm
from enum import Enum


def get_rank():
    """Get the rank of the current process."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))


def put_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    dst_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Put to a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            -1 by default, which means local copy.
        unroll_factor: int
            The unroll factor
        enable_aggressive_vectorize: bool
            Whether to enable aggressive vectorization.
            If True, the compiler with try to vectorize the copy via int4.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.put"), src, dst, size, dst_pe, unroll_factor, "warp", enable_aggressive_vectorize
    )


def get_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    src_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Get from a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            -1 by default, which means local copy.
        unroll_factor: int
            The unroll factor
        enable_aggressive_vectorize: bool
            Whether to enable aggressive vectorization.
            If True, the compiler with try to vectorize the copy via int4.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.get"), src, dst, size, src_pe, unroll_factor, "warp", enable_aggressive_vectorize
    )


def put_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, dst_pe: PrimExpr | IntImm | None = -1):
    """Put to a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            -1 by default, which means local copy.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.put"), src, dst, size, dst_pe, 0, "block", True
    )  # NOTE: unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | IntImm | None = -1):
    """Get from a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            -1 by default, which means local copy.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.get"), src, dst, size, src_pe, 0, "block", True
    )  # NOTE: unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy


class BinaryRelation(Enum):
    EQ = 0
    NE = 1
    GE = 2
    LE = 3
    GT = 4
    LT = 5


def wait_eq(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value == expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.EQ.value, address_of(value), expected, peer)


def wait_ne(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value != expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.NE.value, address_of(value), expected, peer)


def wait_ge(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value >= expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.GE.value, address_of(value), expected, peer)


def wait_le(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value <= expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.LE.value, address_of(value), expected, peer)


def wait_gt(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value > expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.GT.value, address_of(value), expected, peer)


def wait_lt(value: PrimExpr, expected: PrimExpr, peer: PrimExpr | None = -1):
    """Wait until value < expected"""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.wait"), BinaryRelation.LT.value, address_of(value), expected, peer)
