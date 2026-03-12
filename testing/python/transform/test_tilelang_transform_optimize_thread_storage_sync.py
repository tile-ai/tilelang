# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tvm.tir.stmt_functor import post_order_visit


def _count_shared_syncs(func) -> int:
    count = 0

    def _visit(node):
        nonlocal count
        if not (isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op)):
            return
        if str(node.op.name) != "tir.tvm_storage_sync" or len(node.args) != 1:
            return
        arg = node.args[0]
        if isinstance(arg, tvm.tir.StringImm) and arg.value in {"shared", "shared.dyn"}:
            count += 1

    post_order_visit(func.body, _visit)
    return count


def _count_waits(func) -> int:
    count = 0

    def _visit(node):
        nonlocal count
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tir.ptx_wait_group":
            count += 1

    post_order_visit(func.body, _visit)
    return count


def _run(mod):
    mod = tl.transform.ThreadSync("shared")(mod)
    mod = tl.transform.Simplify()(mod)
    return mod


def test_optimize_thread_storage_sync_removes_shadowed_sync():
    @T.prim_func
    def before(A: T.Tensor((8,), T.uint8), B: T.Tensor((8,), T.uint8)):
        S = T.alloc_buffer((8,), dtype=T.uint8, scope="shared")
        T.tvm_storage_sync("shared")
        T.ptx_cp_async(T.access_ptr(S[0], "w", 4), T.access_ptr(A[0], "r", 4), 4)
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        B[0] = S[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)

    assert _count_shared_syncs(mod["main"]) == 1
    assert _count_waits(mod["main"]) == 1


def test_optimize_thread_storage_sync_keeps_sync_before_shared_consume():
    @T.prim_func
    def before(A: T.Tensor((8,), T.uint8), B: T.Tensor((8,), T.uint8)):
        S = T.alloc_buffer((8,), dtype=T.uint8, scope="shared")
        T.tvm_storage_sync("shared")
        B[0] = S[0]
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        B[1] = A[1]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)

    assert _count_shared_syncs(mod["main"]) == 2
    assert _count_waits(mod["main"]) == 1


def test_optimize_thread_storage_sync_keeps_sync_before_opaque_shared_call():
    @T.prim_func
    def before(A: T.Tensor((8,), T.uint8)):
        S = T.alloc_buffer((8,), dtype=T.uint8, scope="shared")
        T.tvm_storage_sync("shared")
        T.evaluate(
            T.call_extern(
                "handle",
                "opaque_shared_touch",
                T.tvm_access_ptr(
                    T.type_annotation("uint8"),
                    S.data,
                    0,
                    4,
                    2,
                ),
            )
        )
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        A[0] = T.uint8(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)

    assert _count_shared_syncs(mod["main"]) == 2
    assert _count_waits(mod["main"]) == 1


def test_optimize_thread_storage_sync_removes_preheader_sync_covered_by_loop_entry():
    @T.prim_func
    def before(A: T.Tensor((8,), T.uint8), B: T.Tensor((8,), T.uint8)):
        S = T.alloc_buffer((8,), dtype=T.uint8, scope="shared")
        T.tvm_storage_sync("shared")
        for i in T.serial(0, 4):
            for j in T.unroll(0, 2):
                B[j] = T.uint8(1)
            T.ptx_cp_async(T.access_ptr(S[0], "w", 4), T.access_ptr(A[0], "r", 4), 4)
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.tvm_storage_sync("shared")
            B[2] = S[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)

    assert _count_shared_syncs(mod["main"]) == 1
    assert _count_waits(mod["main"]) == 1
