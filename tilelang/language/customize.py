"""Some customized operations frequently used in tensor programming, exposed on the TileLang language surface."""

from __future__ import annotations
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, op
from tilelang.utils.language import bits_product, prim_expr_equal
from .atomic import atomic_max, atomic_min, atomic_add, atomic_addx2, atomic_addx4, atomic_load, atomic_store  # noqa: F401


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]

    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def reshape(src: Buffer, shape: list[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.

    Args:
        src (Buffer): Input buffer to be reshaped
        shape (List[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    assert prim_expr_equal(bits_product(shape, src.dtype), bits_product(src.shape, src.dtype)), (
        f"T.reshape/view shape check failed. src {src} src.shape: {src.shape}, src.dtype: {src.dtype}, target shape: {shape}, target dtype: {src.dtype}"
    )
    return T.Tensor(shape, src.dtype, src.data)


def view(src: Buffer, shape: list[PrimExpr] | None = None, dtype: str | None = None) -> Buffer:
    """Return a Tensor view of the input buffer with an optional new shape and dtype.

    If `shape` is None the source buffer's shape is used; if `dtype` is None the source buffer's dtype is used. The returned buffer shares the same underlying data as `src` (no copy).
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    assert prim_expr_equal(bits_product(shape, dtype), bits_product(src.shape, src.dtype)), "T.reshape/view shape check failed."
    return T.Tensor(shape, dtype, src.data)


def loop_break():
    """Break out of the current loop.

    Returns:
        tir.Call: A call to the `tl.loop_break` intrinsic.
    """
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))
