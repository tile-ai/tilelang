# ruff: noqa
import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls
from tilelang._compat.tensor import TensorView


TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
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
def test_zero_width_tmem_reports_value_error():
    tensor = TensorView.create("malformed_tensor", [128, 32], "void", scope="shared.tmem")
    func = tvm.tirx.PrimFunc([], tensor.allocate()).with_attr("target", TARGET)
    mod = tvm.IRModule.from_expr(func)

    with pytest.raises(ValueError, match="TMEM allocation requires a positive element bit width"):
        tl.cuda.transform.LowerSharedTmem()(mod)


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_only_suppresses_matching_buffer()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
    test_zero_width_tmem_reports_value_error()
