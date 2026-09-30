"""PTODSL tracing helpers for TileLang PTO SIMT codegen."""

from __future__ import annotations

import ptodsl._allreduce as _allreduce
import ptodsl._scalar as _scalar
from ptodsl import pto
from ptodsl._scalar import _emit_llvm_byte_pointer
from ptodsl._surface_values import VecValue, unwrap_surface_value, wrap_surface_value, resolve_address_access
from ptodsl._types import _DType, _integer_signedness, _restore_integer_signedness, _signless_integer_type, _strip_integer_signedness
from ptoas.mlir.dialects import llvm
from ptoas.mlir.ir import IntegerType


def _vector_lane(value, index):
    raw_index = unwrap_surface_value(pto.const(index, dtype=pto.i32))
    return wrap_surface_value(llvm.ExtractElementOp(unwrap_surface_value(value), raw_index).res)


def shuffle_vec(dtype: _DType, values: list | tuple, indices: list[int] | tuple[int, ...]):
    """Pick lanes by constant index from concatenated scalar/vector inputs."""
    entries = [(value, value.size) if isinstance(value, VecValue) else (value, 1) for value in values]
    picked = []
    for index in indices:
        lane = index
        for value, size in entries:
            if lane < size:
                picked.append(_vector_lane(value, lane) if size > 1 else value)
                break
            lane -= size
        else:
            raise IndexError(f"shuffle_vec index {index} is outside the concatenated inputs")
    if len(picked) == 1:
        return picked[0]
    return vector_from_list(dtype, picked)


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


def _redux_signedness(type_obj):
    # Maps signed ints to signless pto.iXX (uint keeps pto.uiXX).
    return "unsigned" if _integer_signedness(type_obj) == "unsigned" else "signed"


def _redux_integer_compat(op, value, *, with_signedness):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return op(value)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    if with_signedness:
        result = op(signless_value, signedness=_redux_signedness(raw_value.type))
    else:
        result = op(signless_value)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


def _redux_add(value):
    # redux_add takes no signedness argument.
    return _redux_integer_compat(pto.redux_add, value, with_signedness=False)


def _redux_max(value):
    # redux_max/min require an explicit integer signedness.
    return _redux_integer_compat(pto.redux_max, value, with_signedness=True)


def _redux_min(value):
    return _redux_integer_compat(pto.redux_min, value, with_signedness=True)


def _shuffle_bfly(value, offset):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.shuffle_bfly(value, offset)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.shuffle_bfly(signless_value, offset)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


# PTOAS's redux/shuffle take signless i32 carriers while its allreduce dispatches on signed/unsigned types.
# Run the si32 strategy on the signless carrier with identity constants typed to match,
# instead of reinterpreting to si32 and back.
_allreduce._REDUCER_REDUX.update(
    {
        "sum": _redux_add,
        "max": _redux_max,
        "min": _redux_min,
    }
)
_allreduce.shuffle_bfly = _shuffle_bfly
_allreduce._REDUCER_IDENTITY_DTYPE["si32"] = pto.i32

_scalar_select = _scalar.select


def _scalar_select_compat(cond, true_val, false_val):
    # PTOAS's select drops the authored signedness of integer operands, which
    # breaks e.g. uint32 allreduce branch merges; restore it.
    authored_type = unwrap_surface_value(true_val).type
    result = _scalar_select(cond, true_val, false_val)
    raw_result = unwrap_surface_value(result)
    if IntegerType.isinstance(authored_type) and _signless_integer_type(authored_type) == raw_result.type:
        result = wrap_surface_value(_restore_integer_signedness(raw_result, authored_type))
    return result


_scalar.select = _scalar_select_compat

_simt_allreduce = _allreduce._simt_allreduce


def _simt_allreduce_compat(value, *, threads, scale=1, thread_offset=0, scratch=None, reducer):
    """Dispatch signless i32 through PTOAS's si32 strategy; other dtypes delegate."""
    raw_value = unwrap_surface_value(value)
    if not (
        IntegerType.isinstance(raw_value.type)
        and _integer_signedness(raw_value.type) == "signless"
        and IntegerType(raw_value.type).width == 32
    ):
        return _simt_allreduce(
            value,
            threads=threads,
            scale=scale,
            thread_offset=thread_offset,
            scratch=scratch,
            reducer=reducer,
        )
    _allreduce._check_params(threads=threads, scale=scale, thread_offset=thread_offset)
    if threads <= scale:
        return value
    args = dict(dtype="si32", threads=threads, scale=scale, thread_offset=thread_offset, reducer=reducer)
    if threads <= 32 and _allreduce._is_pow2(threads) and _allreduce._is_pow2(scale):
        return _allreduce._emit_warp_reduce(value, **args)
    if scratch is None:
        raise ValueError(f"all_reduce {reducer}/si32/t{threads}/s{scale}/o{thread_offset} requires a UB scratch buffer")
    _allreduce._validate_scratch_buffer(
        scratch,
        value_type=raw_value.type,
        reducer=reducer,
        dtype="si32",
        threads=threads,
        scale=scale,
        thread_offset=thread_offset,
    )
    if threads <= 32:
        return _allreduce._emit_ub_reduce(value, scratch, **args)
    if scale <= 32 and _allreduce._is_pow2(threads) and _allreduce._is_pow2(scale):
        return _allreduce._emit_cross_warp_reduce(value, scratch, **args)
    return _allreduce._emit_ub_reduce(value, scratch, **args)


def simt_allreduce_sum(value, **kwargs):
    return _simt_allreduce_compat(value, reducer="sum", **kwargs)


def simt_allreduce_max(value, **kwargs):
    return _simt_allreduce_compat(value, reducer="max", **kwargs)


def simt_allreduce_min(value, **kwargs):
    return _simt_allreduce_compat(value, reducer="min", **kwargs)
