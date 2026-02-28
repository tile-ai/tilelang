from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm.tir.stmt_functor import post_order_visit


def _count_calls(func):
    call_count = {}

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            key = str(node.op.name)
            call_count[key] = call_count.get(key, 0) + 1

    post_order_visit(func.body, _visit)
    return call_count


def _collect_wait_args(func):
    wait_args = []

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            if str(node.op.name) == "tir.ptx_wait_group" and len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, tvm.tir.IntImm):
                    wait_args.append(int(arg.value))

    post_order_visit(func.body, _visit)
    return wait_args


def _run(mod):
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.OptimizeCPAsyncSync()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.OptimizeCPAsyncSync()(mod)
    mod = tl.transform.Simplify()(mod)
    return mod


def test_optimize_cp_async_sync_removes_redundant_commit():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.serial(0, 4):
            T.ptx_cp_async(
                T.access_ptr(S[i * 4], "w", 4),
                T.access_ptr(A[i * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    calls = _count_calls(mod["main"])
    assert calls.get("tir.ptx_commit_group", 0) == 1


def test_optimize_cp_async_sync_removes_weaker_wait():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.serial(0, 4):
            T.ptx_cp_async(
                T.access_ptr(S[i * 4], "w", 4),
                T.access_ptr(A[i * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.ptx_wait_group(1)
            B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    calls = _count_calls(mod["main"])
    assert calls.get("tir.ptx_wait_group", 0) == 1


def test_optimize_cp_async_sync_keeps_stricter_wait():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.serial(0, 4):
            T.ptx_cp_async(
                T.access_ptr(S[i * 4], "w", 4),
                T.access_ptr(A[i * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(1)
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    calls = _count_calls(mod["main"])
    assert calls.get("tir.ptx_wait_group", 0) == 2


def test_optimize_cp_async_sync_relaxes_loop_wait_with_prefetch():
    @T.prim_func
    def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8)):
        S = T.alloc_buffer((32,), dtype=T.uint8, scope="shared")
        # Prologue prefetch.
        T.ptx_cp_async(T.access_ptr(S[0], "w", 4), T.access_ptr(A[0], "r", 4), 4)
        T.ptx_commit_group()
        for i in T.serial(0, 4):
            T.ptx_cp_async(
                T.access_ptr(S[(i + 1) * 4], "w", 4),
                T.access_ptr(A[(i + 1) * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]
        T.ptx_wait_group(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    wait_args = _collect_wait_args(mod["main"])
    assert 1 in wait_args, f"Expected a relaxed wait_group(1), got wait args {wait_args}"


def test_optimize_cp_async_sync_does_not_relax_wait_without_prefetch():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.serial(0, 4):
            T.ptx_cp_async(
                T.access_ptr(S[i * 4], "w", 4),
                T.access_ptr(A[i * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    wait_args = _collect_wait_args(mod["main"])
    assert 1 not in wait_args, f"Did not expect wait_group(1) without prefetch, got wait args {wait_args}"


if __name__ == "__main__":
    tilelang.testing.main()
