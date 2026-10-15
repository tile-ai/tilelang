from __future__ import annotations

from numbers import Integral
from typing import Any

import tvm
from tvm.ir import PrimExpr
from tvm.ir.base import Span
from tvm.runtime import const
from tvm.tirx import Buffer
from tvm.tirx.expr import IntImm, Shuffle as Shuffle
import tvm.tirx.op as _tvm_op

from tilelang.language.dtypes import _is_any_dtype
from tilelang.utils.deprecated import deprecated_warning


def _buffer_data(value):
    if isinstance(value, Buffer):
        return value.data
    return value


def _normalize_primexpr_args(args):
    return tuple(_buffer_data(arg) for arg in args)


# Re-export unchanged TVM operators without adding another Python call layer.
# TileLang-specific adapters remain as functions below.
call_packed = _tvm_op.call_packed
call_cpacked = _tvm_op.call_cpacked
call_packed_lowered = _tvm_op.call_packed_lowered
call_cpacked_lowered = _tvm_op.call_cpacked_lowered
call_llvm_intrin = _tvm_op.call_llvm_intrin
call_llvm_pure_intrin = _tvm_op.call_llvm_pure_intrin
tvm_stack_alloca = _tvm_op.tvm_stack_alloca
tvm_stack_make_shape = _tvm_op.tvm_stack_make_shape
tvm_stack_make_array = _tvm_op.tvm_stack_make_array
assume = _tvm_op.assume
undef = _tvm_op.undef
start_profile_intrinsic = _tvm_op.start_profile_intrinsic
end_profile_intrinsic = _tvm_op.end_profile_intrinsic
tvm_tuple = _tvm_op.tvm_tuple
tvm_struct_get = _tvm_op.tvm_struct_get
tvm_struct_set = _tvm_op.tvm_struct_set
address_of = _tvm_op.address_of
lookup_param = _tvm_op.lookup_param
tvm_thread_allreduce = _tvm_op.tvm_thread_allreduce
tvm_thread_invariant = _tvm_op.tvm_thread_invariant
tvm_storage_sync = _tvm_op.tvm_storage_sync
tvm_warp_shuffle = _tvm_op.tvm_warp_shuffle
tvm_warp_shuffle_up = _tvm_op.tvm_warp_shuffle_up
tvm_warp_shuffle_down = _tvm_op.tvm_warp_shuffle_down
tvm_warp_activemask = _tvm_op.tvm_warp_activemask
tvm_throw_last_error = _tvm_op.tvm_throw_last_error
tvm_load_matrix_sync = _tvm_op.tvm_load_matrix_sync
tvm_mma_sync = _tvm_op.tvm_mma_sync
tvm_bmma_sync = _tvm_op.tvm_bmma_sync
tvm_fill_fragment = _tvm_op.tvm_fill_fragment
tvm_store_matrix_sync = _tvm_op.tvm_store_matrix_sync
ptx_commit_group = _tvm_op.ptx_commit_group
ptx_wait_group = _tvm_op.ptx_wait_group
ptx_cp_async_barrier = _tvm_op.ptx_cp_async_barrier
ptx_init_barrier_thread_count = _tvm_op.ptx_init_barrier_thread_count
ptx_arrive_barrier = _tvm_op.ptx_arrive_barrier
create_barriers = _tvm_op.create_barriers
vectorlow = _tvm_op.vectorlow
vectorhigh = _tvm_op.vectorhigh
vectorcombine = _tvm_op.vectorcombine
ret = _tvm_op.ret
min_value = _tvm_op.min_value
max_value = _tvm_op.max_value
exp = _tvm_op.exp
exp2 = _tvm_op.exp2
exp10 = _tvm_op.exp10
erf = _tvm_op.erf
tanh = _tvm_op.tanh
sigmoid = _tvm_op.sigmoid
log = _tvm_op.log
log2 = _tvm_op.log2
log10 = _tvm_op.log10
log1p = _tvm_op.log1p
tan = _tvm_op.tan
cos = _tvm_op.cos
cosh = _tvm_op.cosh
acos = _tvm_op.acos
acosh = _tvm_op.acosh
sin = _tvm_op.sin
sinh = _tvm_op.sinh
asin = _tvm_op.asin
asinh = _tvm_op.asinh
atan = _tvm_op.atan
atanh = _tvm_op.atanh
atan2 = _tvm_op.atan2
sqrt = _tvm_op.sqrt
rsqrt = _tvm_op.rsqrt
clz = _tvm_op.clz
floor = _tvm_op.floor
ceil = _tvm_op.ceil
trunc = _tvm_op.trunc
abs = _tvm_op.abs
bitwise_and = _tvm_op.bitwise_and
bitwise_not = _tvm_op.bitwise_not
bitwise_or = _tvm_op.bitwise_or
bitwise_xor = _tvm_op.bitwise_xor
nearbyint = _tvm_op.nearbyint
nextafter = _tvm_op.nextafter
hypot = _tvm_op.hypot
copysign = _tvm_op.copysign
ldexp = _tvm_op.ldexp
likely = _tvm_op.likely
isnan = _tvm_op.isnan
isnullptr = _tvm_op.isnullptr
isfinite = _tvm_op.isfinite
isinf = _tvm_op.isinf
popcount = _tvm_op.popcount
q_multiply_shift = _tvm_op.q_multiply_shift
q_multiply_shift_per_axis = _tvm_op.q_multiply_shift_per_axis
shift_left = _tvm_op.shift_left
shift_right = _tvm_op.shift_right
fmod = _tvm_op.fmod
if_then_else = _tvm_op.if_then_else
truncdiv = _tvm_op.truncdiv
truncmod = _tvm_op.truncmod
floordiv = _tvm_op.floordiv
floormod = _tvm_op.floormod
ceildiv = _tvm_op.ceildiv
TVMBackendAllocWorkspace = _tvm_op.TVMBackendAllocWorkspace
TVMBackendFreeWorkspace = _tvm_op.TVMBackendFreeWorkspace
anylist_getitem = _tvm_op.anylist_getitem
anylist_resetitem = _tvm_op.anylist_resetitem
anylist_setitem_call_packed = _tvm_op.anylist_setitem_call_packed
anylist_setitem_call_cpacked = _tvm_op.anylist_setitem_call_cpacked
vscale = _tvm_op.vscale


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

    lanes = vector.dtype.lanes
    if lanes <= 1:
        raise ValueError(f"extract_lane expects a vector expression, but got dtype {vector.dtype}")

    if isinstance(lane, IntImm):
        lane = lane.value
    elif not isinstance(lane, Integral):
        raise TypeError(f"extract_lane expects a compile-time integer lane, but got {type(lane).__name__}")

    lane = int(lane)
    if lane < 0 or lane >= lanes:
        raise IndexError(f"Lane index {lane} is out of bounds for dtype {vector.dtype} with {lanes} lanes")

    return Shuffle([vector], [lane], span)


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
    return _tvm_op.ptx_arrive_barrier_expect_tx(barrier_id, byte_count)


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
    return call_intrin(dtype, _tvm_op.Op.get("tl.infinity"), tvm.tirx.StringImm(str(dtype)), span=span)


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
    return _tvm_op.reinterpret(dtype, value, span)


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
        return _tvm_op.round(x, span)
    if rounding_mode == "ties-away-from-zero":
        x = tvm.tirx.convert(x)
        return call_intrin(x.dtype, _tvm_op.Op.get("tl.round_ties_away_from_zero"), x, span=span)
    raise ValueError(f"Unsupported T.round rounding_mode {rounding_mode!r}; expected 'ties-to-even' or 'ties-away-from-zero'.")


def pow_of_int(x: PrimExpr, y: int) -> PrimExpr:
    """Fast power operation than pow(float, float).

    Args:
        x (PrimExpr): Base value
        y (int): Exponent value
    """
    return call_intrin(
        x.dtype,
        tvm.tirx.op.Op.get("tl.pow_of_int"),
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
            return const(1, dtype=x.dtype)
        if yv >= 1:
            return pow_of_int(x, yv)
        return _tvm_op.pow(x, yv, span)
    return _tvm_op.pow(x, y, span)


# pylint: disable=unnecessary-lambda
