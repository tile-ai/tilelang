"""PTODSL tracing helpers for TileLang PTO SIMT codegen."""

from __future__ import annotations

import ptodsl._allreduce as _allreduce
from ptodsl import pto
from ptodsl._scalar import _emit_llvm_byte_pointer
from ptodsl._surface_values import VecValue, unwrap_surface_value, wrap_surface_value, resolve_address_access
from ptodsl._types import _DType, _integer_signedness, _restore_integer_signedness, _strip_integer_signedness
from ptoas.mlir.dialects import llvm
from ptoas.mlir.ir import IntegerType


def _vector_lane(value, index):
    raw_index = unwrap_surface_value(pto.const(index, dtype=pto.i32))
    return wrap_surface_value(llvm.ExtractElementOp(unwrap_surface_value(value), raw_index).res)


def vector_from_list(dtype: _DType, values: list | tuple) -> VecValue:
    """Build a PTODSL builtin vector from scalar values in a Python sequence."""
    values = tuple(values)
    return pto.Vec(dtype, len(values), init=values)


def vector_to_list(value: VecValue) -> list:
    """Extract all lanes of a PTODSL builtin vector into a Python list."""
    return [_vector_lane(value, lane) for lane in range(value.size)]


def store_vector_to_list(dst: list, offset: int, value: VecValue) -> None:
    """Store all lanes of a PTODSL vector into a Python-list local buffer."""
    for lane, scalar in enumerate(vector_to_list(value)):
        dst[offset + lane] = scalar


def vectorize_unary_f32x2(op, value):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(value, 0)),
            op(_vector_lane(value, 1)),
        ),
    )


def vectorize_binary_f32x2(op, lhs, rhs):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(lhs, 0), _vector_lane(rhs, 0)),
            op(_vector_lane(lhs, 1), _vector_lane(rhs, 1)),
        ),
    )


def vectorize_binary_fp8(op, lhs, rhs, dtype):
    """Compute FP8 vectors through the supported packed two-lane conversions."""
    if op not in {"+", "-", "*"}:
        raise ValueError(f"unsupported FP8 binary operation: {op}")
    if lhs.size != rhs.size or lhs.size not in {2, 4, 8}:
        raise ValueError("FP8 binary operands must have equal 2, 4, or 8 lanes")

    def compute_pair(a, b):
        a = pto.cast(a, pto.f32, rounding="r", saturation="nosat")
        b = pto.cast(b, pto.f32, rounding="r", saturation="nosat")
        value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()
        return pto.cast(value, dtype, rounding="r", saturation="sat")

    if lhs.size == 2:
        return compute_pair(lhs, rhs)

    # FP8 builtin vectors cannot be split with llvm.extractelement. Local
    # storage allows the same bit pattern to be reloaded as supported x2 pairs.
    lhs_local = pto.alloc_buffer((lhs.size,), dtype)
    rhs_local = pto.alloc_buffer((rhs.size,), dtype)
    out_local = pto.alloc_buffer((lhs.size,), dtype)
    pto.store(lhs, lhs_local, 0, contiguous=lhs.size)
    pto.store(rhs, rhs_local, 0, contiguous=rhs.size)
    for offset in range(0, lhs.size, 2):
        a = pto.load(lhs_local, offset, contiguous=2)
        b = pto.load(rhs_local, offset, contiguous=2)
        pto.store(compute_pair(a, b), out_local, offset, contiguous=2)
    return pto.load(out_local, 0, contiguous=lhs.size)


def fp8_byte_load(buffer, index, dtype):
    """Load one FP8 storage byte without materializing an unsupported scalar FP8."""
    address, offset = resolve_address_access(buffer, index)
    pointer = _emit_llvm_byte_pointer(address, offset, dtype.resolve())
    return wrap_surface_value(llvm.LoadOp(IntegerType.get_signless(8), pointer).res)


def fp8_byte_store(value, buffer, index, dtype):
    """Store one FP8 storage byte through its integer representation."""
    address, offset = resolve_address_access(buffer, index)
    pointer = _emit_llvm_byte_pointer(address, offset, dtype.resolve())
    llvm.StoreOp(unwrap_surface_value(value), pointer)


def scalar_binary_fp8(op, lhs, rhs, dtype):
    """Use the packed FP8 conversion primitive for one logical element."""
    if op not in {"+", "-", "*"}:
        raise ValueError(f"unsupported FP8 binary operation: {op}")
    a_local = pto.alloc_buffer((2,), dtype)
    b_local = pto.alloc_buffer((2,), dtype)
    out_local = pto.alloc_buffer((2,), dtype)
    for index in range(2):
        fp8_byte_store(lhs, a_local, index, dtype)
        fp8_byte_store(rhs, b_local, index, dtype)
    a = pto.load(a_local, 0, contiguous=2)
    b = pto.load(b_local, 0, contiguous=2)
    a = pto.cast(a, pto.f32, rounding="r", saturation="nosat")
    b = pto.cast(b, pto.f32, rounding="r", saturation="nosat")
    value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()
    result = pto.cast(value, dtype, rounding="r", saturation="sat")
    pto.store(result, out_local, 0, contiguous=2)
    return fp8_byte_load(out_local, 0, dtype)


def scalar_div(lhs, rhs):
    return lhs / rhs


def scalar_rsqrt(value):
    return 1.0 / pto.sqrt(value)


def _redux_integer_compat(op, value):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return op(value)
    signedness = _integer_signedness(raw_value.type)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = op(signless_value, signedness=signedness)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


def _redux_add(value):
    return _redux_integer_compat(pto.redux_add, value)


def _redux_max(value):
    return _redux_integer_compat(pto.redux_max, value)


def _redux_min(value):
    return _redux_integer_compat(pto.redux_min, value)


def _shuffle_bfly(value, offset):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.shuffle_bfly(value, offset)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.shuffle_bfly(signless_value, offset)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


# PTOAS v0.1.6 models integer signedness on PTODSL values, while its low-level
# redux and shuffle operations require signless LLVM-compatible i32 carriers.
# Keep the public allreduce implementation and adapt only those two boundaries.
_allreduce._REDUCER_REDUX.update(
    {
        "sum": _redux_add,
        "max": _redux_max,
        "min": _redux_min,
    }
)
_allreduce.shuffle_bfly = _shuffle_bfly


def simt_allreduce_sum(value, **kwargs):
    return _allreduce.simt_allreduce_sum(value, **kwargs)


def simt_allreduce_max(value, **kwargs):
    return _allreduce.simt_allreduce_max(value, **kwargs)


def simt_allreduce_min(value, **kwargs):
    return _allreduce.simt_allreduce_min(value, **kwargs)
