import tvm.script.ir_builder.tir.ir as _ir
from tvm.script.ir_builder.tir import frame
from tvm.tir import PrimExpr
from typing import Any
import tilelang.language.tir.op as _tir_op
import functools


def serial(start: PrimExpr, stop: PrimExpr = None, *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.serial(start=start, stop=stop, annotations=annotations)


def parallel(start: PrimExpr, stop: PrimExpr = None, *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.parallel(start=start, stop=stop, annotations=annotations)


def vectorized(start: PrimExpr, stop: PrimExpr = None, *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.vectorized(start=start, stop=stop, annotations=annotations)


def unroll(start: PrimExpr, stop: PrimExpr = None, *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    # Ensure annotations has {"pragma_unroll_explicit": True} by default
    if annotations is None:
        annotations = {"pragma_unroll_explicit": False}
    else:
        # Add "pragma_unroll_explicit": True if not already present
        annotations = dict(annotations)
        annotations.setdefault("pragma_unroll_explicit", False)
    return _ir.unroll(start=start, stop=stop, annotations=annotations)


def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    *,
    annotations: dict[str, Any] = None,
) -> frame.ForFrame:
    """The thread-binding For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    thread : str
        The thread for loop variable to bind.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.thread_binding(start=start, stop=stop, thread=thread, annotations=annotations)


def grid(*extents: PrimExpr) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ir.grid(*extents)


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)

    return wrapped


def _op_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)

    return wrapped


abs = _op_wrapper(_tir_op.abs)  # pylint: disable=redefined-builtin
acos = _op_wrapper(_tir_op.acos)
acosh = _op_wrapper(_tir_op.acosh)
address_of = _op_wrapper(_tir_op.address_of)
asin = _op_wrapper(_tir_op.asin)
asinh = _op_wrapper(_tir_op.asinh)
atan = _op_wrapper(_tir_op.atan)
atan2 = _op_wrapper(_tir_op.atan2)
atanh = _op_wrapper(_tir_op.atanh)
bitwise_and = _op_wrapper(_tir_op.bitwise_and)
bitwise_not = _op_wrapper(_tir_op.bitwise_not)
bitwise_or = _op_wrapper(_tir_op.bitwise_or)
bitwise_xor = _op_wrapper(_tir_op.bitwise_xor)
ceil = _op_wrapper(_tir_op.ceil)
clz = _op_wrapper(_tir_op.clz)
copysign = _op_wrapper(_tir_op.copysign)
cos = _op_wrapper(_tir_op.cos)
cosh = _op_wrapper(_tir_op.cosh)
erf = _op_wrapper(_tir_op.erf)
exp = _op_wrapper(_tir_op.exp)
exp2 = _op_wrapper(_tir_op.exp2)
exp10 = _op_wrapper(_tir_op.exp10)
floor = _op_wrapper(_tir_op.floor)
ceildiv = _op_wrapper(_tir_op.ceildiv)
floordiv = _op_wrapper(_tir_op.floordiv)
floormod = _op_wrapper(_tir_op.floormod)
fmod = _op_wrapper(_tir_op.fmod)
hypot = _op_wrapper(_tir_op.hypot)
if_then_else = _op_wrapper(_tir_op.if_then_else)
infinity = _op_wrapper(_tir_op.infinity)
isfinite = _op_wrapper(_tir_op.isfinite)
isinf = _op_wrapper(_tir_op.isinf)
isnan = _op_wrapper(_tir_op.isnan)
isnullptr = _op_wrapper(_tir_op.isnullptr)
ldexp = _op_wrapper(_tir_op.ldexp)
likely = _op_wrapper(_tir_op.likely)
log = _op_wrapper(_tir_op.log)
log1p = _op_wrapper(_tir_op.log1p)
log2 = _op_wrapper(_tir_op.log2)
log10 = _op_wrapper(_tir_op.log10)
lookup_param = _op_wrapper(_tir_op.lookup_param)
max_value = _op_wrapper(_tir_op.max_value)
min_value = _op_wrapper(_tir_op.min_value)
nearbyint = _op_wrapper(_tir_op.nearbyint)
nextafter = _op_wrapper(_tir_op.nextafter)
popcount = _op_wrapper(_tir_op.popcount)
pow = _op_wrapper(_tir_op.pow)  # pylint: disable=redefined-builtin
q_multiply_shift = _op_wrapper(_tir_op.q_multiply_shift)
q_multiply_shift_per_axis = _op_wrapper(_tir_op.q_multiply_shift_per_axis)
ret = _op_wrapper(_tir_op.ret)
round = _op_wrapper(_tir_op.round)  # pylint: disable=redefined-builtin
rsqrt = _op_wrapper(_tir_op.rsqrt)
shift_left = _op_wrapper(_tir_op.shift_left)
shift_right = _op_wrapper(_tir_op.shift_right)
sigmoid = _op_wrapper(_tir_op.sigmoid)
sin = _op_wrapper(_tir_op.sin)
sinh = _op_wrapper(_tir_op.sinh)
sqrt = _op_wrapper(_tir_op.sqrt)
tan = _op_wrapper(_tir_op.tan)
tanh = _op_wrapper(_tir_op.tanh)
trunc = _op_wrapper(_tir_op.trunc)
truncdiv = _op_wrapper(_tir_op.truncdiv)
truncmod = _op_wrapper(_tir_op.truncmod)
tvm_access_ptr = _op_wrapper(_tir_op.tvm_access_ptr)
tvm_throw_last_error = _op_wrapper(_tir_op.tvm_throw_last_error)
tvm_stack_alloca = _op_wrapper(_tir_op.tvm_stack_alloca)
tvm_stack_make_shape = _op_wrapper(_tir_op.tvm_stack_make_shape)
tvm_stack_make_array = _op_wrapper(_tir_op.tvm_stack_make_array)
tvm_check_return = _op_wrapper(_tir_op.tvm_check_return)
call_packed = _op_wrapper(_tir_op.call_packed)
call_cpacked = _op_wrapper(_tir_op.call_cpacked)
call_packed_lowered = _op_wrapper(_tir_op.call_packed_lowered)
call_cpacked_lowered = _op_wrapper(_tir_op.call_cpacked_lowered)
tvm_tuple = _op_wrapper(_tir_op.tvm_tuple)
tvm_struct_set = _op_wrapper(_tir_op.tvm_struct_set)
tvm_struct_get = _tir_op.tvm_struct_get
tvm_thread_invariant = _op_wrapper(_tir_op.tvm_thread_invariant)
tvm_thread_allreduce = _op_wrapper(_tir_op.tvm_thread_allreduce)
tvm_load_matrix_sync = _op_wrapper(_tir_op.tvm_load_matrix_sync)
tvm_mma_sync = _op_wrapper(_tir_op.tvm_mma_sync)
tvm_bmma_sync = _op_wrapper(_tir_op.tvm_bmma_sync)
tvm_fill_fragment = _op_wrapper(_tir_op.tvm_fill_fragment)
tvm_store_matrix_sync = _op_wrapper(_tir_op.tvm_store_matrix_sync)
tvm_storage_sync = _tir_op.tvm_storage_sync
tvm_warp_shuffle = _tir_op.tvm_warp_shuffle
tvm_warp_shuffle_up = _tir_op.tvm_warp_shuffle_up
tvm_warp_shuffle_down = _tir_op.tvm_warp_shuffle_down
tvm_warp_activemask = _tir_op.tvm_warp_activemask
ptx_wait_group = _op_wrapper(_tir_op.ptx_wait_group)
ptx_commit_group = _op_wrapper(_tir_op.ptx_commit_group)
ptx_cp_async_barrier = _op_wrapper(_tir_op.ptx_cp_async_barrier)
ptx_init_barrier_thread_count = _op_wrapper(_tir_op.ptx_init_barrier_thread_count)
ptx_arrive_barrier = _op_wrapper(_tir_op.ptx_arrive_barrier)
ptx_arrive_barrier_expect_tx = _op_wrapper(_tir_op.ptx_arrive_barrier_expect_tx)
ptx_wait_barrier = _op_wrapper(_tir_op.ptx_wait_barrier)
create_barriers = _op_wrapper(_tir_op.create_barriers)
assume = _op_wrapper(_tir_op.assume)
undef = _op_wrapper(_tir_op.undef)
TVMBackendAllocWorkspace = _op_wrapper(_tir_op.TVMBackendAllocWorkspace)
TVMBackendFreeWorkspace = _op_wrapper(_tir_op.TVMBackendFreeWorkspace)
start_profile_intrinsic = _op_wrapper(_tir_op.start_profile_intrinsic)
end_profile_intrinsic = _op_wrapper(_tir_op.end_profile_intrinsic)
anylist_getitem = _op_wrapper(_tir_op.anylist_getitem)
anylist_resetitem = _op_wrapper(_tir_op.anylist_resetitem)
anylist_setitem_call_packed = _op_wrapper(_tir_op.anylist_setitem_call_packed)
anylist_setitem_call_cpacked = _op_wrapper(_tir_op.anylist_setitem_call_cpacked)
vscale = _op_wrapper(_tir_op.vscale)

reinterpret = _dtype_forward(_tir_op.reinterpret)
call_extern = _dtype_forward(_tir_op.call_extern)
call_intrin = _dtype_forward(_tir_op.call_intrin)
call_llvm_intrin = _dtype_forward(_tir_op.call_llvm_intrin)
call_llvm_pure_intrin = _dtype_forward(_tir_op.call_llvm_pure_intrin)
call_pure_extern = _dtype_forward(_tir_op.call_pure_extern)
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tir_op.ptx_mma_sp)
ptx_wgmma_ss = _dtype_forward(_tir_op.ptx_wgmma_ss)
ptx_wgmma_rs = _dtype_forward(_tir_op.ptx_wgmma_rs)
ptx_tcgen05_mma_ss = _dtype_forward(_tir_op.ptx_tcgen05_mma_ss)
ptx_tcgen05_mma_ts = _dtype_forward(_tir_op.ptx_tcgen05_mma_ts)
ptx_ldmatrix = _dtype_forward(_tir_op.ptx_ldmatrix)
ptx_cp_async = _dtype_forward(_tir_op.ptx_cp_async)
ptx_cp_async_bulk = _dtype_forward(_tir_op.ptx_cp_async_bulk)
mma_store = _dtype_forward(_tir_op.mma_store)
mma_fill = _dtype_forward(_tir_op.mma_fill)
vectorlow = _dtype_forward(_tir_op.vectorlow)
vectorhigh = _dtype_forward(_tir_op.vectorhigh)
vectorcombine = _dtype_forward(_tir_op.vectorcombine)
tvm_mfma = _dtype_forward(_tir_op.tvm_mfma)
tvm_mfma_store = _dtype_forward(_tir_op.tvm_mfma_store)
tvm_rdna_wmma = _dtype_forward(_tir_op.tvm_rdna_wmma)
tvm_rdna_wmma_store = _dtype_forward(_tir_op.tvm_rdna_wmma_store)
