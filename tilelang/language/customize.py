"""Some customized operations frequently used in tensor programming, exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang.language import _construction
from tilelang.language import _tensor_spec

from tilelang._typing import ShapeType, DType, BufferLikeType
import tilelang.language as T
from tvm_ffi import dtype as DataType
from tvm_ffi._dtype import DataTypeCode
from tilelang._analyzer import create_analyzer
from tvm_tirx.ir import Expr as PrimExpr
from tilelang.language._tensor_spec import TensorValue as Buffer
from tvm_tirx.ir.prim import Broadcast
from tilelang.utils.language import bits_product, prim_expr_equal, retrieve_buffer_and_offset
from .atomic import atomic_max, atomic_min, atomic_add, atomic_addx2, atomic_addx4, atomic_load, atomic_or, atomic_store  # noqa: F401


def dp4a(A: BufferLikeType, B: BufferLikeType, C: BufferLikeType) -> PrimExpr:
    """Perform a four-element signed int8 dot product accumulated into int32.

    Args:
        A: First int8 input buffer.
        B: Second int8 input buffer.
        C: Int32 accumulator buffer.

    Returns:
        Native void expression for the DP4A operation.

    Raises:
        ValueError: If A or B is not int8, or C is not int32.
    """
    a_dtype = T.dtype(_construction.dtype_of(retrieve_buffer_and_offset(A)[0]))
    b_dtype = T.dtype(_construction.dtype_of(retrieve_buffer_and_offset(B)[0]))
    c_dtype = T.dtype(_construction.dtype_of(retrieve_buffer_and_offset(C)[0]))
    if a_dtype != T.int8:
        raise ValueError(f"dp4a requires int8 inputs, got A.dtype='{a_dtype}'")
    if b_dtype != T.int8:
        raise ValueError(f"dp4a requires int8 inputs, got B.dtype='{b_dtype}'")
    if c_dtype != T.int32:
        raise ValueError(f"dp4a requires an int32 accumulator, got C.dtype='{c_dtype}'")
    return T.call_extern(
        "void",
        "DP4A",
        T.access_ptr(A, "r", extent=4),
        T.access_ptr(B, "r", extent=4),
        T.access_ptr(C, "rw", extent=1),
    )


_NON_FLOAT_TYPE_CODES = (
    DataTypeCode.INT,
    DataTypeCode.UINT,
    DataTypeCode.BOOL,
    DataTypeCode.HANDLE,
)


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]

    Floating-point ``NaN`` values in the input or either bound propagate to
    the result. When ``min_val > max_val``, the result is ``max_val`` (unless
    an operand is ``NaN``), matching ``torch.clamp``. Each operand is evaluated
    once, and vector operands are clamped independently in each lane.

    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Value clamped to the specified range
    """
    dst, min_val, max_val = (_construction.as_expr(value) for value in (dst, min_val, max_val))
    clamped = T.min(T.max(dst, min_val), max_val)
    dtype = DataType(_construction.dtype_of(clamped))
    if dtype.type_code in _NON_FLOAT_TYPE_CODES:
        return clamped
    # Match min/max's type promotion, including scalar bounds on vector inputs.
    lanes = max(DataType(_construction.dtype_of(value)).lanes for value in (dst, min_val, max_val))
    dtype = dtype.with_lanes(lanes)
    args = []
    for value in (dst, min_val, max_val):
        if DataType(_construction.dtype_of(value)).lanes == 1 and lanes != 1:
            value = Broadcast(value, lanes)
        args.append(T.cast(value, dtype))
    return T.call_intrin(dtype, "tl.clamp", *args)


def reshape(src: Buffer, shape: ShapeType) -> Buffer:
    """Reshapes the input buffer to the specified shape.

    Args:
        src (Buffer): Input buffer to be reshaped
        shape (ShapeType): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    bits, src_bits = bits_product(shape, _construction.dtype_of(src)), bits_product(src.shape, _construction.dtype_of(src))
    assert prim_expr_equal(bits, src_bits) or create_analyzer().can_prove_equal(bits, src_bits), (
        f"T.reshape/view shape check failed. {bits_product(shape, _construction.dtype_of(src))}, {bits_product(src.shape, _construction.dtype_of(src))}"
    )
    return T.Tensor(shape, _construction.dtype_of(src), src.data)


def view(
    src: Buffer,
    shape: ShapeType | None = None,
    dtype: DType | None = None,
    strides: ShapeType | None = None,
) -> Buffer:
    """Return a Tensor view with an optional new shape, dtype, and explicit strides.

    If ``shape`` is None, the source buffer's shape is used. If ``dtype`` is None,
    the source buffer's dtype is used. The returned buffer shares its data with
    ``src`` without copying.

    Without ``strides``, ``src`` must be densely packed and the result uses a dense
    layout. A non-contiguous source requires explicit ``strides``. Explicit strides are
    expressed in units of ``dtype`` elements and allow rank changes.

    The source must have zero ``elem_offset``. The caller must guarantee that explicit
    strides are valid for the backing storage, including dtype alignment and physical
    contiguity between any source dimensions merged by a rank-changing view.
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = _construction.dtype_of(src)

    analyzer = create_analyzer()

    # A view must describe the same logical storage starting at the same address.
    view_bits = bits_product(shape, dtype)
    src_bits = bits_product(src.shape, _construction.dtype_of(src))
    if not (prim_expr_equal(view_bits, src_bits) or bool(analyzer.can_prove_equal(view_bits, src_bits))):
        raise ValueError(
            f"T.view shape and dtype must preserve the logical bit count of '{src.name}': {view_bits} bits != {src_bits} bits."
        )
    if not (prim_expr_equal(_tensor_spec.tensor_offset(src), 0) or bool(analyzer.can_prove_equal(_tensor_spec.tensor_offset(src), 0))):
        raise ValueError("T.view does not support a source buffer with non-zero elem_offset.")

    if strides is None:
        # Only an already-compact source may use the implicit dense layout.
        if len(_tensor_spec.tensor_strides(src)) != 0:
            if len(_tensor_spec.tensor_strides(src)) != len(src.shape):
                raise ValueError(f"T.view requires explicit strides for non-contiguous buffer '{src.name}'.")
            expected_stride = 1
            for extent, stride in zip(reversed(list(src.shape)), reversed(list(_tensor_spec.tensor_strides(src)))):
                extent_is_singleton = prim_expr_equal(extent, 1) or bool(analyzer.can_prove_equal(extent, 1))
                stride_is_compact = prim_expr_equal(stride, expected_stride) or bool(analyzer.can_prove_equal(stride, expected_stride))
                if not (extent_is_singleton or stride_is_compact):
                    raise ValueError(f"T.view requires explicit strides for non-contiguous buffer '{src.name}'.")
                expected_stride = expected_stride * extent
        dense_strides = []
        stride = 1
        for extent in reversed(list(shape)):
            dense_strides.append(stride)
            stride = stride * extent
        return T.StridedTensor(shape, tuple(reversed(dense_strides)), dtype, data=src.data)

    # Explicit strides are expressed in destination elements.
    if len(strides) != len(shape):
        raise ValueError(f"T.view expected {len(shape)} strides for shape {tuple(shape)}, but got {len(strides)}.")
    for dim, stride in enumerate(strides):
        is_negative = stride < 0 if isinstance(stride, int) else analyzer.can_prove(stride < 0)
        if is_negative:
            raise ValueError(f"T.view stride at dim {dim} must be non-negative, but got {stride}.")

    return T.StridedTensor(shape, tuple(strides), dtype, data=src.data)


def loop_break():
    """Emit a native break from the innermost loop in the active builder."""
    return _construction.break_loop()
