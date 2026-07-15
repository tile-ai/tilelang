"""Some customized operations frequently used in tensor programming, exposed on the TileLang language surface."""

from __future__ import annotations
from tilelang._typing import ShapeType, DType, BufferLikeType
import tilelang.language as T
from tvm import arith
from tvm.tirx import PrimExpr, Buffer, op
from tilelang.utils.language import bits_product, prim_expr_equal, retrieve_buffer_and_offset
from .atomic import atomic_max, atomic_min, atomic_add, atomic_addx2, atomic_addx4, atomic_load, atomic_or, atomic_store  # noqa: F401


def dp4a(A: BufferLikeType, B: BufferLikeType, C: BufferLikeType) -> PrimExpr:
    """Perform a four-element signed int8 dot product accumulated into int32.

    Args:
        A: First int8 input buffer.
        B: Second int8 input buffer.
        C: Int32 accumulator buffer.

    Returns:
        Handle to the DP4A operation.

    Raises:
        ValueError: If A or B is not int8, or C is not int32.
    """
    a_dtype = T.dtype(retrieve_buffer_and_offset(A)[0].dtype)
    b_dtype = T.dtype(retrieve_buffer_and_offset(B)[0].dtype)
    c_dtype = T.dtype(retrieve_buffer_and_offset(C)[0].dtype)
    if a_dtype != T.int8:
        raise ValueError(f"dp4a requires int8 inputs, got A.dtype='{a_dtype}'")
    if b_dtype != T.int8:
        raise ValueError(f"dp4a requires int8 inputs, got B.dtype='{b_dtype}'")
    if c_dtype != T.int32:
        raise ValueError(f"dp4a requires an int32 accumulator, got C.dtype='{c_dtype}'")
    return T.call_extern(
        "handle",
        "DP4A",
        T.access_ptr(A, "r"),
        T.access_ptr(B, "r"),
        T.access_ptr(C, "rw"),
    )


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


def reshape(src: Buffer, shape: ShapeType) -> Buffer:
    """Reshapes the input buffer to the specified shape.

    Args:
        src (Buffer): Input buffer to be reshaped
        shape (ShapeType): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    bits, src_bits = bits_product(shape, src.dtype), bits_product(src.shape, src.dtype)
    assert prim_expr_equal(bits, src_bits) or arith.Analyzer().can_prove_equal(bits, src_bits), (
        f"T.reshape/view shape check failed. {bits_product(shape, src.dtype)}, {bits_product(src.shape, src.dtype)}"
    )
    return T.Tensor(shape, src.dtype, src.data)


def view(src: Buffer, shape: ShapeType | None = None, dtype: DType | None = None) -> Buffer:
    """Return a Tensor view of the input buffer with an optional new shape and dtype.

    If `shape` is None the source buffer's shape is used; if `dtype` is None the source buffer's dtype is used. The returned buffer shares the same underlying data as `src` (no copy).
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    bits, src_bits = bits_product(shape, dtype), bits_product(src.shape, src.dtype)
    assert prim_expr_equal(bits, src_bits) or arith.Analyzer().can_prove_equal(bits, src_bits), (
        f"T.reshape/view shape check failed. {bits_product(shape, dtype)}, {bits_product(src.shape, src.dtype)}"
    )
    return T.Tensor(shape, dtype, src.data)


def loop_break() -> PrimExpr:
    """Break out of the current loop.

    Returns:
        tir.Call: A call to the `tl.loop_break` intrinsic.
    """
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))
