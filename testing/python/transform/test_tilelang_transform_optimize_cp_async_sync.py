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
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and str(node.op.name) == "tir.ptx_wait_group"
            and len(node.args) == 1
        ):
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


def test_optimize_cp_async_sync_relaxes_loop_head_wait_with_prefetch():
    @T.prim_func
    def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8)):
        S = T.alloc_buffer((32,), dtype=T.uint8, scope="shared")
        # Prologue prefetch: keep two committed groups in flight.
        T.ptx_cp_async(T.access_ptr(S[0], "w", 4), T.access_ptr(A[0], "r", 4), 4)
        T.ptx_commit_group()
        T.ptx_cp_async(T.access_ptr(S[4], "w", 4), T.access_ptr(A[4], "r", 4), 4)
        T.ptx_commit_group()
        for i in T.serial(0, 4):
            # Leading wait inserted by pipelining.
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]
            # Prefetch for i+2.
            T.ptx_cp_async(
                T.access_ptr(S[(i + 2) * 4], "w", 4),
                T.access_ptr(A[(i + 2) * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
        T.ptx_wait_group(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    wait_args = _collect_wait_args(mod["main"])
    assert 1 in wait_args, f"Expected a relaxed wait_group(1), got wait args {wait_args}"


def test_optimize_cp_async_sync_does_not_relax_loop_head_wait_without_prefetch():
    @T.prim_func
    def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8)):
        S = T.alloc_buffer((32,), dtype=T.uint8, scope="shared")
        # Only one committed group before the loop; relaxing would be unsafe.
        T.ptx_cp_async(T.access_ptr(S[0], "w", 4), T.access_ptr(A[0], "r", 4), 4)
        T.ptx_commit_group()
        for i in T.serial(0, 4):
            T.ptx_wait_group(0)
            B[i * 4] = S[i * 4]
            T.ptx_cp_async(
                T.access_ptr(S[(i + 1) * 4], "w", 4),
                T.access_ptr(A[(i + 1) * 4], "r", 4),
                4,
            )
            T.ptx_commit_group()
        T.ptx_wait_group(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)
    wait_args = _collect_wait_args(mod["main"])
    assert 1 not in wait_args, f"Did not expect relaxed wait_group(1), got wait args {wait_args}"


def test_optimize_cp_async_sync_splits_epilogue_wait_between_two_consumer_phases():
    def _is_wait_stmt(stmt, wait_n: int) -> bool:
        if not isinstance(stmt, tvm.tir.Evaluate):
            return False
        call = stmt.value
        if not (isinstance(call, tvm.tir.Call) and isinstance(call.op, tvm.ir.Op)):
            return False
        if str(call.op.name) != "tir.ptx_wait_group" or len(call.args) != 1:
            return False
        arg = call.args[0]
        return isinstance(arg, tvm.tir.IntImm) and int(arg.value) == wait_n

    def _is_shared_storage_sync(stmt) -> bool:
        if not isinstance(stmt, tvm.tir.Evaluate):
            return False
        call = stmt.value
        if not (isinstance(call, tvm.tir.Call) and isinstance(call.op, tvm.ir.Op)):
            return False
        if str(call.op.name) != "tir.tvm_storage_sync" or len(call.args) != 1:
            return False
        arg = call.args[0]
        return isinstance(arg, tvm.tir.StringImm) and arg.value == "shared"

    @T.prim_func
    def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8)):
        S = T.alloc_buffer((32,), dtype=T.uint8, scope="shared")
        for i in T.serial(0, 2):
            T.ptx_cp_async(T.access_ptr(S[i * 4], "w", 4), T.access_ptr(A[i * 4], "r", 4), 4)
            T.ptx_commit_group()
        # Epilogue drain inserted by pipelining.
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        # First epilogue consumer phase.
        B[0] = S[0]
        # Barrier between consumer phases.
        T.tvm_storage_sync("shared")
        # Second epilogue consumer phase.
        B[1] = S[4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = _run(mod)

    wait_args = _collect_wait_args(mod["main"])
    assert 1 in wait_args, f"Expected wait_group(1) after splitting epilogue, got {wait_args}"
    assert 0 in wait_args, f"Expected an inserted wait_group(0) for the final drain, got {wait_args}"

    def _unwrap_to_seq(stmt):
        while True:
            if isinstance(stmt, tvm.tir.SeqStmt):
                return stmt
            if isinstance(stmt, tvm.tir.Allocate):
                stmt = stmt.body
                continue
            if isinstance(stmt, tvm.tir.AllocateConst):
                stmt = stmt.body
                continue
            if isinstance(stmt, tvm.tir.DeclBuffer):
                stmt = stmt.body
                continue
            if isinstance(stmt, tvm.tir.LetStmt):
                stmt = stmt.body
                continue
            if isinstance(stmt, tvm.tir.AttrStmt):
                stmt = stmt.body
                continue
            if isinstance(stmt, tvm.tir.BlockRealize):
                stmt = stmt.block.body
                continue
            if isinstance(stmt, tvm.tir.Block):
                stmt = stmt.body
                continue
            return None

    top_seq = _unwrap_to_seq(mod["main"].body)
    assert top_seq is not None, f"Expected a SeqStmt after unwrapping, got:\n{mod['main']}"
    seq = list(top_seq.seq)

    # The original post-loop wait_group(0) should be relaxed to wait_group(1).
    wait1_idx = next((i for i, s in enumerate(seq) if _is_wait_stmt(s, 1)), None)
    assert wait1_idx is not None, f"Expected a top-level wait_group(1), got:\n{mod['main']}"
    assert wait1_idx + 1 < len(seq) and _is_shared_storage_sync(
        seq[wait1_idx + 1]
    ), "Expected tvm_storage_sync('shared') immediately after relaxed wait_group(1)"

    store_indices = [i for i, s in enumerate(seq) if isinstance(s, tvm.tir.BufferStore)]
    store_indices = [i for i in store_indices if i > wait1_idx]
    assert len(store_indices) >= 2, f"Expected two global BufferStore statements, got indices {store_indices}"
    first_store, second_store = store_indices[0], store_indices[1]

    split_sync_idx = next(
        (i for i in range(first_store + 1, second_store) if _is_shared_storage_sync(seq[i])),
        None,
    )
    assert split_sync_idx is not None, "Expected a shared barrier between the two global stores"
    assert split_sync_idx - 1 >= 0 and _is_wait_stmt(
        seq[split_sync_idx - 1], 0
    ), "Expected an inserted wait_group(0) immediately before the barrier between epilogue blocks"


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
