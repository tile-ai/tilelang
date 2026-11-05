# ruff: noqa
import pytest

import tvm_tirx
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls
from tvm_tirx import ir, tirx


TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100"})


def _apply(func):
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm_tirx.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.cuda.transform.LowerSharedTmem()(mod)
    return mod


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_explicit_deallocate_tmem_suppresses_auto_dealloc():
    """Explicit T.deallocate_tmem on fallthrough suppresses auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            T.deallocate_tmem(C_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert_call_count(body, op="tl.ptx_init_tensor_memory", count=1)
    assert_call_count(body, op="tl.ptx_deallocate_tensor_memory", count=1)
    assert_call_count(body, op="tl.deallocate_tmem", count=0)

    dealloc_call = collect_calls(body, op="tl.ptx_deallocate_tensor_memory")[0]
    assert dealloc_call.args[1].value == 128


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_explicit_deallocate_only_suppresses_matching_buffer():
    """Only the explicitly-deallocated buffer skips auto-dealloc; others keep it."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem([128, 128], T.float32)
            B_tmem = T.alloc_tmem([128, 64], T.float32)
            T.deallocate_tmem(A_tmem)

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = collect_calls(body, op="tl.ptx_deallocate_tensor_memory")
    # A_tmem: 1 explicit (auto suppressed); B_tmem: 1 auto = 2 total
    assert len(dealloc_calls) == 2

    dealloc_num_cols = sorted(call.args[1].value for call in dealloc_calls)
    assert dealloc_num_cols == [64, 128]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_dealloc_before_thread_return_keeps_auto_dealloc():
    """Dealloc on non-fallthrough path (before thread_return) does NOT suppress auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            tx = T.get_thread_binding()

            if tx < 32:
                T.deallocate_tmem(C_tmem)
                T.thread_return()

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = collect_calls(body, op="tl.ptx_deallocate_tensor_memory")
    # 1 explicit (non-fallthrough) + 1 auto (block end) = 2
    assert len(dealloc_calls) == 2
    assert [call.args[1].value for call in dealloc_calls] == [128, 128]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_untyped_tensor_producer_reports_type_error():
    # Native tensors carry their storage type directly. The old Buffer with an
    # independently untyped data Var is no longer constructible; an incorrectly
    # typed producer must instead be rejected by the native call verifier.
    tensor_type = tirx.TensorType((128, 1), "float32", scope="shared.tmem")
    malformed = ir.Call("tirx.alloc_tensor", [], ty=ir.PrimType("int64"), ty_args=[tensor_type])
    with pytest.raises(TypeError, match="Tensor producer result must match its TensorType argument"):
        malformed.validate()


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_only_suppresses_matching_buffer()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
    test_untyped_tensor_producer_reports_type_error()
