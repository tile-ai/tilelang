# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


TARGET = tvm.target.Target("cuda -arch=sm_100")


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.LowerSharedTmem()(mod)
    return mod


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def test_explicit_deallocate_tmem_suppresses_auto_dealloc():
    """Explicit T.deallocate_tmem on fallthrough suppresses auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            T.deallocate_tmem(C_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, "tl.ptx_init_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.ptx_deallocate_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    dealloc_call = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")[0]
    assert dealloc_call.args[1].value == 128


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

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # A_tmem: 1 explicit (auto suppressed); B_tmem: 1 auto = 2 total
    assert len(dealloc_calls) == 2

    dealloc_num_cols = sorted(call.args[1].value for call in dealloc_calls)
    assert dealloc_num_cols == [64, 128]


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

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # 1 explicit (non-fallthrough) + 1 auto (block end) = 2
    assert len(dealloc_calls) == 2
    assert [call.args[1].value for call in dealloc_calls] == [128, 128]


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_only_suppresses_matching_buffer()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
