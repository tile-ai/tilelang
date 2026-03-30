# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.LowerSharedBarrier()(mod)
    return mod


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def _collect_storage_syncs(stmt):
    return _collect_calls(stmt, "tir.tvm_storage_sync")


def _collect_init_barrier_calls(stmt):
    return _collect_calls(stmt, "tir.ptx_init_barrier_thread_count")


def _collect_fence_barrier_init(stmt):
    return _collect_calls(stmt, "tl.ptx_fence_barrier_init")


def _collect_shuffle_elect(stmt):
    return _collect_calls(stmt, "tl.tl_shuffle_elect")


def test_single_barrier():
    """Single barrier with one arrive count."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            mbar = T.alloc_barrier(128)  # noqa: F841

    mod = _apply(func)
    body = mod["main"].body

    assert len(_collect_init_barrier_calls(body)) == 1
    assert len(_collect_fence_barrier_init(body)) == 1
    assert len(_collect_shuffle_elect(body)) == 1

    init_call = _collect_init_barrier_calls(body)[0]
    # arrive count should be 128
    assert init_call.args[1].value == 128


def test_multiple_barriers():
    """Multiple barriers with different arrive counts."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            mbars = T.alloc_barrier([1, 1, 128, 128])  # noqa: F841

    mod = _apply(func)
    body = mod["main"].body

    init_calls = _collect_init_barrier_calls(body)
    assert len(init_calls) == 4

    arrive_counts = sorted([c.args[1].value for c in init_calls])
    assert arrive_counts == [1, 1, 128, 128]

    # Should have exactly one shuffle_elect guard and one fence_barrier_init
    assert len(_collect_shuffle_elect(body)) == 1
    assert len(_collect_fence_barrier_init(body)) == 1

    # Should sync after init
    syncs = _collect_storage_syncs(body)
    assert len(syncs) >= 1


def test_no_barrier_is_noop():
    """Pass should be a no-op when no barrier buffers are present."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            buf = T.alloc_shared((16,), T.float16)
            buf[0] = T.float16(0)

    mod = _apply(func)
    body = mod["main"].body

    assert len(_collect_init_barrier_calls(body)) == 0
    assert len(_collect_fence_barrier_init(body)) == 0


if __name__ == "__main__":
    tilelang.testing.main()
