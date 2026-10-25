"""TileLang intrinsic spellings lowered to native typed TIRx operations."""

from tvm_tirx import ir, tirx
from tvm_tirx.backend.cuda import op as cuda
from tvm_tirx.backend.cuda.ptx import PTXNamespace
from tvm_tirx.tirx import op

_ptx = PTXNamespace()


def _pointer(value):
    if tirx.is_tensor_var(value) or isinstance(value, ir.TensorLoad):
        return op.address_of(value)
    return value


def ptx_arrive_barrier(barrier):
    return _ptx["mbarrier.arrive.shared.b64"](_pointer(barrier))


def ptx_arrive_barrier_expect_tx(barrier, byte_count):
    return _ptx["mbarrier.arrive.expect_tx.shared.b64"](_pointer(barrier), byte_count)


def ptx_init_barrier_thread_count(barrier, count):
    return _ptx["mbarrier.init.shared.b64"](_pointer(barrier), count)


def ptx_cp_async_barrier(barrier):
    return _ptx["cp.async.mbarrier.arrive.shared.b64"](_pointer(barrier))


def ptx_commit_group():
    return _ptx["cp.async.commit_group"]()


def ptx_wait_group(count):
    return _ptx["cp.async.wait_group"](count)


def ptx_wait_barrier(barrier, phase):
    return cuda.mbarrier_wait(_pointer(barrier), phase)


def ptx_mma(dtype, *args):
    return cuda.ptx_legacy_mma(*args, ty=ir.PrimType(dtype))


def ptx_mma_sp(dtype, *args):
    return ir.Call("tl.ptx_mma_sp", list(args), ty="void")


def mma_store(dtype, *args):
    return cuda.mma_store(*args, ty=ir.PrimType(dtype))


def ptx_cp_async_bulk(dtype, dst, src, size, barrier):
    return _ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](_pointer(dst), _pointer(src), size, _pointer(barrier))


def vectorlow(dtype, value):
    return op.vector_low(value, ty=ir.PrimType(dtype))


def vectorhigh(dtype, value):
    return op.vector_high(value, ty=ir.PrimType(dtype))


def vectorcombine(dtype, first, second):
    return op.vector_combine(first, second, ty=ir.PrimType(dtype))


_ALIASES = {
    "tvm_stack_alloca": op.stack_alloca,
    "tvm_stack_make_shape": op.stack_make_shape,
    "tvm_stack_make_array": op.stack_make_dltensor,
    "tvm_thread_allreduce": op.gpu_thread_allreduce,
    "tvm_thread_invariant": op.gpu_thread_invariant,
    "tvm_storage_sync": op.gpu_storage_sync,
    "tvm_load_matrix_sync": op.gpu_load_matrix_sync,
    "tvm_mma_sync": op.gpu_mma_sync,
    "tvm_bmma_sync": cuda.bmma_sync,
    "tvm_fill_fragment": op.gpu_fill_fragment,
    "tvm_store_matrix_sync": op.gpu_store_matrix_sync,
    "tvm_warp_shuffle": cuda.__shfl_sync,
    "tvm_warp_shuffle_up": cuda.__shfl_up_sync,
    "tvm_warp_shuffle_down": cuda.__shfl_down_sync,
    "tvm_warp_activemask": cuda.__activemask,
    "TVMBackendAllocWorkspace": op.alloc_workspace,
    "TVMBackendFreeWorkspace": op.free_workspace,
}


def __getattr__(name):
    if name in _ALIASES:
        return _ALIASES[name]
    raise AttributeError(name)
