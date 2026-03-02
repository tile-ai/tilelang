"""Tests for TileLang `LowerPTXAsyncCopy` transform pass."""

from tilelang import tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm.tir.stmt_functor import post_order_visit


def _count_calls(func: tvm.tir.PrimFunc):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(func.body, _visit)
    return counts


def _count_calls_in_stmt(stmt: tvm.tir.Stmt):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(stmt, _visit)
    return counts


def test_lower_ptx_async_copy_rewrites_plain_parallel_copy():
    """LowerPTXAsyncCopy should rewrite plain global->shared stores to cp.async."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.Parallel(16):
            S[i] = A[i]
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0


def test_lower_ptx_async_copy_supports_multi_dim_indices():
    """LowerPTXAsyncCopy should handle N-D buffer indices (pre-FlattenBuffer)."""

    @T.prim_func
    def before(
        A: T.Tensor((4, 4), T.float32),
        B: T.Tensor((4, 4), T.float32),
    ):
        S = T.alloc_buffer((4, 4), dtype=T.float32, scope="shared")
        for i, j in T.Parallel(4, 4):
            S[i, j] = A[i, j]
        B[0, 0] = S[0, 0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0


def test_lower_ptx_async_copy_collapses_vectorized_float16_loop():
    """Vectorized float16 copies should become a single 16B cp.async per loop."""

    @T.prim_func
    def before(
        A: T.Tensor((32,), T.float16),
        B: T.Tensor((32,), T.float16),
    ):
        S = T.alloc_buffer((32,), dtype=T.float16, scope="shared")
        for i in T.serial(4):
            for v in T.vectorized(8):
                S[i * 8 + v] = A[i * 8 + v]
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0


def test_lower_ptx_async_copy_hoists_sync_out_of_predicated_block():
    """Hoist commit+wait out of loops even if block realize predicate != 1."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.serial(4):
            with T.block("copy"):
                vi = T.axis.spatial(4, i)
                T.where(vi < 3)
                S[vi] = A[vi]
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])
    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0

    # Ensure we didn't introduce commit/wait *inside* the serial loop body.
    loop = None

    def _find_for(node):
        nonlocal loop
        if loop is None and isinstance(node, tvm.tir.For):
            loop = node

    post_order_visit(mod["main"].body, _find_for)
    assert loop is not None
    inner_calls = _count_calls_in_stmt(loop.body)
    assert inner_calls.get("tir.ptx_commit_group", 0) == 0
    assert inner_calls.get("tir.ptx_wait_group", 0) == 0


def test_lower_ptx_async_copy_respects_enable_async_copy_config():
    """`tl.enable_async_copy=False` should disable auto rewriting."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.Parallel(16):
            S[i] = A[i]
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    with tvm.transform.PassContext(config={tl.PassConfigKey.TL_ENABLE_ASYNC_COPY: False}):
        mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) == 0
    assert calls.get("tir.ptx_commit_group", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0


def test_lower_ptx_async_copy_does_not_duplicate_existing_sync():
    """If commit/wait already exist, LowerPTXAsyncCopy should not add another pair."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(16):
            S[i] = A[i]
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) == 1
    assert calls.get("tir.ptx_wait_group", 0) == 1


def test_lower_ptx_async_copy_inserts_commit_before_existing_wait():
    """If a wait exists but no commit, we insert a commit to cover injected cp.async."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(16):
            S[i] = A[i]
        T.ptx_wait_group(0)
        B[0] = S[0]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) == 1
    assert calls.get("tir.ptx_wait_group", 0) == 1


def test_lower_ptx_async_copy_keeps_sync_out_of_inner_unrolled_loops_in_pipelined_loop():
    """In a pipelined loop, commit/wait should be top-level statements, not inside copy loops."""

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for ko in T.Pipelined(2, num_stages=2):
            for i in T.unroll(4):
                S[ko * 4 + i] = A[ko * 4 + i]
            B[ko] = S[ko]

    target = tvm.target.Target("cuda -arch=sm_80")
    func = before.with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)

    mod = tl.transform.LowerPTXAsyncCopy()(mod)
    calls = _count_calls(mod["main"])
    assert calls.get("tir.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0

    pipelined_loop = None

    def _find_pipelined_for(node):
        nonlocal pipelined_loop
        if pipelined_loop is None and isinstance(node, tvm.tir.For) and "num_stages" in node.annotations:
            pipelined_loop = node

    post_order_visit(mod["main"].body, _find_pipelined_for)
    assert pipelined_loop is not None

    # Find an inner unrolled loop (the copy loop) and ensure it doesn't contain commit/wait.
    inner_unrolled = None

    def _find_unrolled(node):
        nonlocal inner_unrolled
        if inner_unrolled is None and isinstance(node, tvm.tir.For) and node.kind == tvm.tir.ForKind.UNROLLED:
            inner_unrolled = node

    post_order_visit(pipelined_loop.body, _find_unrolled)
    assert inner_unrolled is not None
    inner_calls = _count_calls_in_stmt(inner_unrolled.body)
    assert inner_calls.get("tir.ptx_commit_group", 0) == 0
    assert inner_calls.get("tir.ptx_wait_group", 0) == 0


if __name__ == "__main__":
    tilelang.testing.main()
