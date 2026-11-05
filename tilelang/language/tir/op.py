from __future__ import annotations
from tilelang.language import _native

from numbers import Integral
from typing import Any

from tilelang.language._native import PrimExpr
from tvm_tirx import ir as _ir
from tvm_tirx.ir import prim as _prim
from tvm_tirx.tirx import op as _tirx_op
from tvm_tirx.ir import Location as Span
from tvm_tirx.ir import PrimType, PtrType, Type, const
from tilelang.language._native import Buffer
from tilelang.language._native import IntImm, Shuffle as Shuffle
from tilelang.language import _native as _tvm_op

from tilelang.language.dtypes import _is_any_dtype, get_tvm_dtype
from tilelang.utils.deprecated import deprecated_warning


def _buffer_data(value):
    if isinstance(value, Buffer):
        return _native.tensor_data_ptr(value)
    return value


def _normalize_primexpr_args(args):
    return tuple(_buffer_data(arg) for arg in args)


# Re-export unchanged TVM operators without adding another Python call layer.
# TileLang-specific adapters remain as functions below.
def call_packed(*args, **kwargs):
    return _tvm_op.call_packed(*args, **kwargs)


def call_cpacked(*args, **kwargs):
    return _tvm_op.call_cpacked(*args, **kwargs)


def call_packed_lowered(*args, **kwargs):
    return _tvm_op.call_packed_lowered(*args, **kwargs)


def call_cpacked_lowered(*args, **kwargs):
    return _tvm_op.call_cpacked_lowered(*args, **kwargs)


def call_llvm_intrin(*args, **kwargs):
    return _tvm_op.call_llvm_intrin(*args, **kwargs)


def call_llvm_pure_intrin(*args, **kwargs):
    return _tvm_op.call_llvm_pure_intrin(*args, **kwargs)


def tvm_stack_alloca(dtype_str, num):
    """Allocate ABI stack storage with the native pointer type for ``dtype_str``."""
    return _tvm_op.stack_alloca(dtype_str, num, loc=_native._loc())


def tvm_stack_make_shape(*args):
    """Build an int64 shape pointer whose storage lives through its packed call."""
    return _tvm_op.stack_make_shape(*args, loc=_native._loc())


def tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset):
    """Build a borrowed DLTensor descriptor; ``elem_offset`` is in elements."""
    return _tvm_op.stack_make_dltensor(_buffer_data(data), shape, strides, ndim, arr_dtype, elem_offset, loc=_native._loc())


def assume(*args, **kwargs):
    return _tvm_op.assume(*args, **kwargs)


def undef(*args, **kwargs):
    return _tvm_op.undef(*args, **kwargs)


def start_profile_intrinsic(id):
    """Construct the legacy profiling marker for the native compiler boundary."""
    return _ir.Call("tl.start_profile_intrinsic", [_native._expr(id)], loc=_native._loc())


def end_profile_intrinsic(id):
    """Construct the end marker paired with :func:`start_profile_intrinsic`."""
    return _ir.Call("tl.end_profile_intrinsic", [_native._expr(id)], loc=_native._loc())


def tvm_tuple(*args, **kwargs):
    return _tvm_op.tvm_tuple(*args, **kwargs)


def tvm_struct_get(*args, **kwargs):
    return _tvm_op.tvm_struct_get(*args, **kwargs)


def tvm_struct_set(*args, **kwargs):
    return _tvm_op.tvm_struct_set(*args, **kwargs)


def address_of(*args, **kwargs):
    return _tvm_op.address_of(*args, **kwargs)


def lookup_param(param_name, span=None):
    """Refer to a named parameter supplied by the C source environment."""
    result = _native.ir.Call("tl.lookup_param", [_native._expr(param_name)], loc=span if span is not None else _native._loc())
    result.validate()
    return result


def tvm_thread_allreduce(*args, **kwargs):
    return _tvm_op.tvm_thread_allreduce(*args, **kwargs)


def tvm_thread_invariant(cond):
    return _tirx_op.gpu_thread_invariant(cond, loc=_native._loc())


def tvm_storage_sync(*args, **kwargs):
    return _tvm_op.tvm_storage_sync(*args, **kwargs)


def tvm_warp_shuffle(mask, value, warp_id, width, warp_size):
    return _tirx_op.gpu_warp_shuffle(mask, value, warp_id, width, warp_size, loc=_native._loc())


def tvm_warp_shuffle_up(mask, value, offset, width, warp_size):
    return _tirx_op.gpu_warp_shuffle_up(mask, value, offset, width, warp_size, loc=_native._loc())


def tvm_warp_shuffle_down(mask, value, offset, width, warp_size):
    return _tirx_op.gpu_warp_shuffle_down(mask, value, offset, width, warp_size, loc=_native._loc())


def tvm_warp_activemask():
    return _tirx_op.gpu_warp_activemask(loc=_native._loc())


def tvm_throw_last_error():
    """Propagate the current FFI error and terminate the host function."""
    return _tvm_op.throw_last_error(loc=_native._loc())


def tvm_load_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
    return _tirx_op.call_intrin(
        "tl.wmma_load_matrix_sync", fragment, m, n, k, index, buffer_ptr, stride, layout, ty=None, loc=_native._loc()
    )


def tvm_mma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c):
    return _tirx_op.gpu_mma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c, loc=_native._loc())


def tvm_bmma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c):
    from tvm_tirx.backend.cuda import op as cuda_op

    return cuda_op.bmma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c, loc=_native._loc())


def tvm_fill_fragment(fragment, m, n, k, index, value):
    return _tirx_op.gpu_fill_fragment(fragment, m, n, k, index, value, loc=_native._loc())


def tvm_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
    return _tirx_op.gpu_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout, loc=_native._loc())


def ptx_commit_group(*args, **kwargs):
    return call_intrin("void", "tl.ptx_commit_group", *args, **kwargs)


def ptx_wait_group(*args, **kwargs):
    return call_intrin("void", "tl.ptx_wait_group", *args, **kwargs)


def ptx_cp_async_barrier(*args, **kwargs):
    return call_intrin("void", "tl.ptx_cp_async_barrier", *args, **kwargs)


def ptx_init_barrier_thread_count(*args, **kwargs):
    return call_intrin("void", "tl.ptx_init_barrier_thread_count", *args, **kwargs)


def ptx_arrive_barrier(*args, **kwargs):
    return call_intrin("void", "tl.ptx_arrive_barrier", *args, **kwargs)


def create_barriers(*args, **kwargs):
    return call_intrin("void", "tl.create_barriers", *args, **kwargs)


def _vector_op(operator, dtype, *vectors):
    result = operator(*[_native._expr(vector) for vector in vectors], ty=None, loc=_native._loc())
    result.validate()
    requested_type = PrimType(get_tvm_dtype(dtype))
    if result.ty != requested_type:
        raise TypeError(f"{operator.__name__} produces {result.ty}, but the requested dtype is {requested_type}")
    return result


def vectorlow(dtype, vec):
    """Return the low half of a vector with the requested result dtype."""
    return _vector_op(_tvm_op.vector_low, dtype, vec)


def vectorhigh(dtype, vec):
    """Return the high half of a vector with the requested result dtype."""
    return _vector_op(_tvm_op.vector_high, dtype, vec)


def vectorcombine(dtype, vec1, vec2):
    """Concatenate equal vector types with the requested result dtype."""
    return _vector_op(_tvm_op.vector_combine, dtype, vec1, vec2)


def ret(val, span=None):
    """Construct a native return statement without emitting it."""
    value = _native._expr(val)
    if not isinstance(value.ty, (PrimType, PtrType)):
        raise TypeError("ret requires a primitive value or pointer")
    return _native.ir.Return(value, loc=span if span is not None else _native._loc())


def min_value(*args, **kwargs):
    return _tvm_op.min_value(*args, **kwargs)


def max_value(*args, **kwargs):
    return _tvm_op.max_value(*args, **kwargs)


def exp(*args, **kwargs):
    return _tvm_op.exp(*args, **kwargs)


def exp2(*args, **kwargs):
    return _tvm_op.exp2(*args, **kwargs)


def exp10(*args, **kwargs):
    return _tvm_op.exp10(*args, **kwargs)


def erf(*args, **kwargs):
    return _tvm_op.erf(*args, **kwargs)


def tanh(*args, **kwargs):
    return _tvm_op.tanh(*args, **kwargs)


def sigmoid(*args, **kwargs):
    return _tvm_op.sigmoid(*args, **kwargs)


def log(*args, **kwargs):
    return _tvm_op.log(*args, **kwargs)


def log2(*args, **kwargs):
    return _tvm_op.log2(*args, **kwargs)


def log10(*args, **kwargs):
    return _tvm_op.log10(*args, **kwargs)


def log1p(*args, **kwargs):
    return _tvm_op.log1p(*args, **kwargs)


def tan(*args, **kwargs):
    return _tvm_op.tan(*args, **kwargs)


def cos(*args, **kwargs):
    return _tvm_op.cos(*args, **kwargs)


def cosh(*args, **kwargs):
    return _tvm_op.cosh(*args, **kwargs)


def acos(*args, **kwargs):
    return _tvm_op.acos(*args, **kwargs)


def acosh(*args, **kwargs):
    return _tvm_op.acosh(*args, **kwargs)


def sin(*args, **kwargs):
    return _tvm_op.sin(*args, **kwargs)


def sinh(*args, **kwargs):
    return _tvm_op.sinh(*args, **kwargs)


def asin(*args, **kwargs):
    return _tvm_op.asin(*args, **kwargs)


def asinh(*args, **kwargs):
    return _tvm_op.asinh(*args, **kwargs)


def atan(*args, **kwargs):
    return _tvm_op.atan(*args, **kwargs)


def atanh(*args, **kwargs):
    return _tvm_op.atanh(*args, **kwargs)


def atan2(*args, **kwargs):
    return _tvm_op.atan2(*args, **kwargs)


def sqrt(*args, **kwargs):
    return _tvm_op.sqrt(*args, **kwargs)


def rsqrt(*args, **kwargs):
    return _tvm_op.rsqrt(*args, **kwargs)


def clz(*args, **kwargs):
    return _tvm_op.clz(*args, **kwargs)


def floor(*args, **kwargs):
    return _tvm_op.floor(*args, **kwargs)


def ceil(*args, **kwargs):
    return _tvm_op.ceil(*args, **kwargs)


def trunc(*args, **kwargs):
    return _tvm_op.trunc(*args, **kwargs)


def abs(*args, **kwargs):
    return _tvm_op.abs(*args, **kwargs)


def bitwise_and(*args, **kwargs):
    return _tvm_op.bitwise_and(*args, **kwargs)


def bitwise_not(*args, **kwargs):
    return _tvm_op.bitwise_not(*args, **kwargs)


def bitwise_or(*args, **kwargs):
    return _tvm_op.bitwise_or(*args, **kwargs)


def bitwise_xor(*args, **kwargs):
    return _tvm_op.bitwise_xor(*args, **kwargs)


def nearbyint(*args, **kwargs):
    return _tvm_op.nearbyint(*args, **kwargs)


def nextafter(*args, **kwargs):
    return _tvm_op.nextafter(*args, **kwargs)


def hypot(*args, **kwargs):
    return _tvm_op.hypot(*args, **kwargs)


def copysign(*args, **kwargs):
    return _tvm_op.copysign(*args, **kwargs)


def ldexp(*args, **kwargs):
    return _tvm_op.ldexp(*args, **kwargs)


def likely(*args, **kwargs):
    return _tvm_op.likely(*args, **kwargs)


def isnan(*args, **kwargs):
    return _tvm_op.isnan(*args, **kwargs)


def isnullptr(*args, **kwargs):
    return _tvm_op.isnullptr(*args, **kwargs)


def isfinite(*args, **kwargs):
    return _tvm_op.isfinite(*args, **kwargs)


def isinf(*args, **kwargs):
    return _tvm_op.isinf(*args, **kwargs)


def popcount(*args, **kwargs):
    return _tvm_op.popcount(*args, **kwargs)


# Quantized lowering is adapted from Apache TVM src/target/intrin_rule.cc
# at 4211874e9e4b8770fd20ad0d98ab80afd1df5028 to native typed IR.
# The Apache-2.0 license and attribution are retained in THIRDPARTYNOTICES.txt.
def _q_expr(value):
    value = _native._expr(value)
    if not isinstance(value.ty, _ir.PrimType):
        raise TypeError("Quantized multiplication expects primitive expressions")
    return value


def _q_dtype(bits, lanes):
    return _ir.PrimType(_ir.PrimType(f"int{bits}").dtype.with_lanes(lanes))


def _q_literal(value):
    if isinstance(value, _ir.IntImm):
        return int(value.value)
    if isinstance(value, _prim.Broadcast) and isinstance(value.value, _ir.IntImm):
        return int(value.value.value)
    return None


def _q_const(value, dtype):
    return _ir.const(value, dtype)


def _q_multiply_shift_common(x, y, q, left_shift, right_shift, is_left_shift_required):
    x, y, q, left_shift, right_shift, is_left_shift_required = map(_q_expr, (x, y, q, left_shift, right_shift, is_left_shift_required))
    for value in (y, left_shift, right_shift):
        if value.ty.dtype.type_code != 0 or value.ty.dtype.bits != 32:
            raise TypeError("Quantized multiplier and shift expressions must be int32")
    lanes = x.ty.dtype.lanes
    hp_dtype = _q_dtype(64, lanes)
    lp_dtype = _q_dtype(32, lanes)
    x = _prim.Cast(hp_dtype, x)
    y = _prim.Cast(hp_dtype, y)
    x = _prim.Select(is_left_shift_required, x << left_shift, x)
    product = x * y
    total_right_shift = right_shift + q
    rounding = _q_const(1, hp_dtype) << (total_right_shift - 1)
    return _prim.Cast(lp_dtype, (product + rounding) >> total_right_shift, loc=_native._loc())


def q_multiply_shift(x, y, q, s):
    """Lower the legacy fixed-point multiplication using native integer IR.

    Round half toward positive infinity, preserving the old 2**30 multiplier
    specialization. The generic path multiplies in int64 and narrows to int32.
    """
    x, y, q, s = map(_q_expr, (x, y, q, s))
    multiplier = _q_literal(y)
    if multiplier is None:
        raise TypeError("q_multiply_shift requires a constant or broadcast-constant multiplier")
    if multiplier == 1 << 30:
        shift = _q_literal(s)
        if shift is None:
            raise TypeError("The power-of-two multiplier path requires a constant shift")
        exponent = shift - 1
        if exponent > 0:
            return _prim.shift_left(x, s - 1, loc=_native._loc())
        # Preserve the old int32 rounding path. Its signed overflow remains
        # outside the old C/LLVM defined domain; do not promise wraparound.
        # s=1 has no valid legacy lowering (it shifts 1 by -1).
        if exponent == 0:
            raise ValueError("q_multiply_shift with multiplier 2**30 and shift 1 has a negative legacy rounding shift")
        exponent_expr = 1 - s
        one = _q_const(1, _q_dtype(32, x.ty.dtype.lanes))
        return _prim.shift_right(x + (one << (exponent_expr - 1)), exponent_expr, loc=_native._loc())
    if s.ty.dtype.type_code != 0 or s.ty.dtype.bits != 32:
        raise TypeError("Quantized shift expression must be int32")
    zero = _q_const(0, s.ty)
    left_shift = _prim.Select(s > zero, s, zero)
    right_shift = _prim.Select(s > zero, zero, -s)
    return _q_multiply_shift_common(x, y, q, left_shift, right_shift, left_shift != zero)


def q_multiply_shift_per_axis(x, y, ls, rs, q, is_lshift_required, is_rshift_required):
    """Lower legacy per-axis quantized multiplication to native integer IR."""
    # Like the old generic legalization, rs itself is authoritative; the final
    # flag is accepted but does not suppress rs and is not evaluated at runtime.
    return _q_multiply_shift_common(x, y, q, ls, rs, is_lshift_required)


def shift_left(*args, **kwargs):
    return _tvm_op.shift_left(*args, **kwargs)


def shift_right(*args, **kwargs):
    return _tvm_op.shift_right(*args, **kwargs)


def fmod(*args, **kwargs):
    return _tvm_op.fmod(*args, **kwargs)


def if_then_else(*args, **kwargs):
    return _tvm_op.if_then_else(*args, **kwargs)


def truncdiv(*args, **kwargs):
    return _tvm_op.truncdiv(*args, **kwargs)


def truncmod(*args, **kwargs):
    return _tvm_op.truncmod(*args, **kwargs)


def floordiv(*args, **kwargs):
    return _tvm_op.floordiv(*args, **kwargs)


def floormod(*args, **kwargs):
    return _tvm_op.floormod(*args, **kwargs)


def ceildiv(*args, **kwargs):
    return _tvm_op.ceildiv(*args, **kwargs)


def TVMBackendAllocWorkspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint):
    """Allocate temporary device workspace; the caller must free the returned pointer."""
    return _tvm_op.alloc_workspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint, loc=_native._loc())


def TVMBackendFreeWorkspace(device_type, device_id, ptr):
    """Free temporary workspace and return the backend's int32 status."""
    return _tvm_op.free_workspace(device_type, device_id, ptr, loc=_native._loc())


def _anylist_call(name, args):
    call = _ir.Call(name, [_native._expr(arg) for arg in args], loc=_native._loc())
    call.validate()
    return call


def anylist_getitem(list_handle, index):
    """Borrow a complete Any slot as a direct packed-call argument."""
    return _anylist_call("tl.anylist_getitem", [list_handle, index])


def anylist_resetitem(list_handle, index):
    """Release the owned value in an Any slot and reset it to None."""
    return _anylist_call("tl.anylist_resetitem", [list_handle, index])


def anylist_setitem_call_packed(list_handle, index, func_name, *args):
    """Move a packed function's complete owned result into an Any slot."""
    return _anylist_call("tl.anylist_setitem_call_packed", [list_handle, index, func_name, *args])


def anylist_setitem_call_cpacked(list_handle, index, func_name, *args):
    """Move a C packed function's complete owned result into an Any slot."""
    return _anylist_call("tl.anylist_setitem_call_cpacked", [list_handle, index, func_name, *args])


def vscale(*args, **kwargs):
    return _tvm_op.vscale(*args, **kwargs)


def extract_lane(vector: PrimExpr, lane: int | IntImm, span: Span | None = None) -> PrimExpr:
    """Extract one scalar lane from a fixed-width vector expression.

    Parameters
    ----------
    vector : PrimExpr
        The vector expression to extract from.

    lane : int or IntImm
        The zero-based lane index. The index must be known at compile time.

    span : Optional[Span]
        The location of this expression in the source code.

    Returns
    -------
    result : PrimExpr
        A scalar expression with the vector's element dtype.
    """
    if not isinstance(vector, PrimExpr):
        raise TypeError(f"extract_lane expects a PrimExpr, but got {type(vector).__name__}")

    lanes = _native.dtype_of(vector).lanes
    if lanes <= 1:
        raise ValueError(f"extract_lane expects a vector expression, but got dtype {_native.dtype_of(vector)}")

    if isinstance(lane, IntImm):
        lane = lane.value
    elif not isinstance(lane, Integral):
        raise TypeError(f"extract_lane expects a compile-time integer lane, but got {type(lane).__name__}")

    lane = int(lane)
    if lane < 0 or lane >= lanes:
        raise IndexError(f"Lane index {lane} is out of bounds for dtype {_native.dtype_of(vector)} with {lanes} lanes")

    return Shuffle([vector], [lane], span if span is not None else _native._loc())


def call_intrin(dtype, func_name, *args, annotations=None, span=None):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = _normalize_primexpr_args(args)
    return _tvm_op.call_intrin(dtype, func_name, *args, annotations=annotations, span=span)


def call_pure_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = _normalize_primexpr_args(args)
    return _tvm_op.call_pure_extern(dtype, func_name, *args, span=span)


def call_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = _normalize_primexpr_args(args)
    return _tvm_op.call_extern(dtype, func_name, *args, span=span)


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Get head access address with memory access pattern info

    Parameters
    ----------
    ptype : Expr
        The data type of pointer.

    data : DType*
        The data of pointer.

    offset : int
        The offset of pointer.

    extent : int
        The extent of pointer.

    rw_mask : int
        The read write mask.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    data = _buffer_data(data)
    return _tvm_op.tvm_access_ptr(ptype, data, offset, extent, rw_mask)


def ptx_cp_async(dst_access_ptr, src_access_ptr, num_elems, predicate=None):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Parameters
    ----------
    dst_access_ptr : PrimExpr
        The destination (shared memory) access pointer created by tvm_access_ptr.
        Should include pointer, offset, extent, and write access flag (rw_mask=2).

    src_access_ptr : PrimExpr
        The source (global memory) access pointer created by tvm_access_ptr.
        Should include pointer, offset, extent, and read access flag (rw_mask=1).

    num_elems : int or PrimExpr
        The number of logical elements to copy.

        For TileLang's ``tl.ptx_cp_async`` frontend op, the final PTX byte width
        is derived later from ``num_elems * element_bits(access_ptr)`` and must
        eventually land on a legal ``cp.async`` width of 4, 8, or 16 bytes.

    predicate : PrimExpr, optional
        Optional predicate condition for conditional cp.async. When provided, the copy
        will only be performed if the predicate evaluates to true. Otherwise, the
        destination will be filled with zeros (default behavior of cp.async).

    Returns
    -------
    call : PrimExpr
        The call expression.

    Examples
    --------
    >>> # Copy 16 uint8 elements (= 16 bytes) from global to shared memory
    >>> T.ptx_cp_async(
    ...     T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),  # dst
    ...     T.tvm_access_ptr(T.type_annotation(T.uint8), B_global.data, 0, 16, 1),  # src
    ...     16  # num_elems
    ... )
    >>>
    >>> # Predicated cp.async (only copy if condition is true)
    >>> T.ptx_cp_async(
    ...     T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
    ...     T.tvm_access_ptr(T.type_annotation(T.uint8), B_global.data, 0, 16, 1),
    ...     16,
    ...     predicate=guard  # only copy if guard is true
    ... )
    """
    if predicate is None:
        return call_intrin("", _tvm_op.Op.get("tl.ptx_cp_async"), dst_access_ptr, src_access_ptr, num_elems)
    else:
        return call_intrin("", _tvm_op.Op.get("tl.ptx_cp_async"), dst_access_ptr, src_access_ptr, num_elems, predicate)


def ptx_arrive_barrier_expect_tx(barrier_id, byte_count):
    """TVM intrinsic for ptx barrier arrival with expect tx using mbarrier.arrive.expect_tx
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    byte_count : int
        Increases the tx count of the mbarrier object to track completion of
        additional async transactions.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("void", "tl.ptx_arrive_barrier_expect_tx", barrier_id, byte_count)


def infinity(dtype: str, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The infinity value of dtype.
    """
    return call_intrin(dtype, _tvm_op.Op.get("tl.infinity"), _native.StringImm(str(dtype)), span=span)


# NOTE(chaofan): Here we use the argument order (value, dtype, ...) instead of (dtype, value, ...) in TVM
# to be consistent with T.cast.
def reinterpret(value, dtype, span: Span | None = None) -> Any:
    """Reinterpret cast a value to dtype.

    Parameters
    ----------
    value : PrimExpr
        The input value.

    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The reinterpret cast value of dtype.
    """

    # NOTE(chaofan): For compatibility, we allow the old API where dtype comes first
    if _is_any_dtype(value):
        deprecated_warning("T.reinterpret(dtype, value)", "reinterpret(value, dtype)")
        value, dtype = dtype, value
    if isinstance(dtype, Type):
        target_type = dtype
    elif str(dtype) == "handle":
        target_type = PtrType("void")
    else:
        target_type = PrimType(get_tvm_dtype(dtype))
    return _tvm_op.reinterpret(_native._expr(value), ty=target_type, loc=span if span is not None else _native._loc())


def round(x, rounding_mode="ties-to-even", span=None):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    rounding_mode : str
        Rounding mode to use. Supported values are ``"ties-to-even"`` and
        ``"ties-away-from-zero"``. ``"ties-to-even"`` is the default and matches
        the existing TileLang/TVM semantics.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    if rounding_mode is None:
        rounding_mode = "ties-to-even"
    elif not isinstance(rounding_mode, str):
        if span is not None:
            raise TypeError("T.round received both a positional span and span=.")
        span = rounding_mode
        rounding_mode = "ties-to-even"

    if rounding_mode == "ties-to-even":
        return _tvm_op.round(_native._expr(x), span if span is not None else _native._loc())
    if rounding_mode == "ties-away-from-zero":
        x = _native.convert(x)
        return call_intrin(_native.dtype_of(x), _tvm_op.Op.get("tl.round_ties_away_from_zero"), x, span=span)
    raise ValueError(f"Unsupported T.round rounding_mode {rounding_mode!r}; expected 'ties-to-even' or 'ties-away-from-zero'.")


def pow_of_int(x: PrimExpr, y: int) -> PrimExpr:
    """Fast power operation than pow(float, float).

    Args:
        x (PrimExpr): Base value
        y (int): Exponent value
    """
    return call_intrin(
        _native.dtype_of(x),
        _native.Op.get("tl.pow_of_int"),
        x,
        y,
    )


def pow(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    if isinstance(y, (int, IntImm)):
        # pow_of_int's `for (i = 1; i < y; ...)` loop only computes x**y for y >= 1.
        yv = int(y)
        if yv == 0:
            return const(1, dtype=_native.dtype_of(x))
        if yv >= 1:
            return pow_of_int(x, yv)
        return _tvm_op.pow(x, yv, span if span is not None else _native._loc())
    return _tvm_op.pow(x, y, span if span is not None else _native._loc())


# pylint: disable=unnecessary-lambda
