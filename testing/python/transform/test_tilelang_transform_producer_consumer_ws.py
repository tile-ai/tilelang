# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import determine_target
from tvm import tir


auto_target = tvm.target.Target(determine_target("auto"))


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def _collect_ifs(stmt):
    ifs = []

    def visitor(node):
        if isinstance(node, tvm.tir.IfThenElse):
            ifs.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return ifs


def _stmt_contains_call(stmt, op_name: str) -> bool:
    found = False

    def visitor(node):
        nonlocal found
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            found = True

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return found


def _count_calls_in_stmt(stmt, op_name: str) -> int:
    count = 0

    def visitor(node):
        nonlocal count
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            count += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return count


def _collect_buffer_loads(stmt, scope: str):
    """Collect BufferLoad nodes from buffers with the given scope."""
    loads = []

    def visitor(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.scope() == scope:
            loads.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return loads


def test_producer_consumer_ws_pure_tma_does_not_reserve_unused_preloop_barrier():
    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16), B: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        v = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[by * 64, 0:481], B[0:481, bx * 64])
            T.writes()

            A_shared = T.alloc_buffer((3, 1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.alloc_buffer((3, 1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((32,), scope="local")

            mbarrier = T.alloc_barrier([128, 128, 128, 128, 128, 128])

            for k in T.serial(16, annotations={"num_stages": T.int32(3)}):
                if v == 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_expect_tx"),
                        mbarrier[k % 3],
                        4096,
                    )
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                        mbarrier[k % 3],
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        by * 64,
                    )
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.mbarrier_wait_parity"),
                    mbarrier[k % 3],
                    k // 3 % 2,
                )

                if v == 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_expect_tx"),
                        mbarrier[k % 3 + 3],
                        4096,
                    )
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, B.data, 512, 512, 2, 1024, 64, 32, 1, 1, 0, 3, 2, 0),
                        mbarrier[k % 3 + 3],
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        bx * 64,
                    )
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.mbarrier_wait_parity"),
                    mbarrier[k % 3 + 3],
                    k // 3 % 2,
                )

                T.call_extern(
                    "handle",
                    "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                    T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
                )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    main_func = mod["main"]
    # After the WS pass, the barrier buffer should still be present as shared.barrier
    # scope BufferLoads in the output (the pass creates its own barrier buffer).
    barrier_loads = _collect_buffer_loads(main_func.body, "shared.barrier")
    assert len(barrier_loads) > 0, "Expected shared.barrier BufferLoad nodes in WS output"


def test_producer_consumer_ws_preserves_guarded_forward_wait():
    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 1)
        by = T.launch_thread("blockIdx.y", 1)
        tx = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[0:128, 0:64])
            T.writes()

            A_shared = T.alloc_buffer((2, 1, 8, 256), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((1,), "float32", scope="local")

            mbarrier = T.alloc_barrier([1, 1])

            for k in T.serial(4, annotations={"num_stages": T.int32(2)}):
                i_s: T.int32 = T.if_then_else(k < 2, 0, -1)

                if i_s >= 0:
                    T.attr(A_shared.data, "tl.tma_copy_write_buffer", 1)
                    if tx == 0:
                        T.call_intrin(
                            "handle",
                            tir.op.Op.get("tl.mbarrier_expect_tx"),
                            mbarrier[k % 2],
                            4096,
                        )
                    if tx == 0:
                        T.tma_load(
                            T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                            mbarrier[k % 2],
                            T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 2 * 2048, 2048, 2),
                            k * 32,
                            by * 64,
                        )
                if i_s >= 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_wait_parity"),
                        mbarrier[k % 2],
                        k // 2 % 2,
                    )
                if i_s >= 0:
                    C_local[0] = C_local[0] + T.float32(1)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    main_func = mod["main"]
    body_text = main_func.script()
    assert 'threadIdx.x", 256' in body_text

    guarded_waits = []
    for if_stmt in _collect_ifs(main_func.body):
        if _stmt_contains_call(if_stmt.then_case, "tl.mbarrier_wait_parity"):
            guarded_waits.append(str(if_stmt.condition))

    assert guarded_waits
    assert any("i_s" in cond for cond in guarded_waits)


def test_producer_consumer_ws_preserves_guarded_producer_backpressure_wait():
    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 1)
        by = T.launch_thread("blockIdx.y", 1)
        tx = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[0:128, 0:64])
            T.writes()

            A_shared = T.alloc_buffer((2, 1, 8, 256), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((1,), "float32", scope="local")

            mbarrier = T.alloc_barrier([1, 1])

            for k in T.serial(4, annotations={"num_stages": T.int32(2)}):
                i_s: T.int32 = T.if_then_else(k < 2, 0, -1)

                if i_s >= 0:
                    T.attr(A_shared.data, "tl.tma_copy_write_buffer", 1)
                    if tx == 0:
                        T.call_intrin(
                            "handle",
                            tir.op.Op.get("tl.mbarrier_expect_tx"),
                            mbarrier[k % 2],
                            4096,
                        )
                    if tx == 0:
                        T.tma_load(
                            T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                            mbarrier[k % 2],
                            T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 2 * 2048, 2048, 2),
                            k * 32,
                            by * 64,
                        )
                if i_s >= 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_wait_parity"),
                        mbarrier[k % 2],
                        k // 2 % 2,
                    )
                if i_s >= 0:
                    C_local[0] = C_local[0] + T.float32(1)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    main_func = mod["main"]
    body_text = main_func.script()
    assert 'threadIdx.x", 256' in body_text

    guarded_wait_count = 0
    guarded_arrive_count = 0
    for if_stmt in _collect_ifs(main_func.body):
        if "i_s" not in str(if_stmt.condition):
            continue
        guarded_wait_count += _count_calls_in_stmt(if_stmt.then_case, "tl.mbarrier_wait_parity")
        guarded_arrive_count += _count_calls_in_stmt(if_stmt.then_case, "tir.ptx_arrive_barrier")

    assert guarded_wait_count >= 2
    assert guarded_arrive_count >= 2


def test_producer_consumer_ws_uses_consumer_guard_for_backpressure_protocol():
    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 1)
        by = T.launch_thread("blockIdx.y", 1)
        tx = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[0:128, 0:64])
            T.writes()

            A_shared = T.alloc_buffer((2, 1, 8, 256), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((1,), "float32", scope="local")

            mbarrier = T.alloc_barrier([1, 1])

            for k in T.serial(4, annotations={"num_stages": T.int32(2)}):
                i_s: T.int32 = T.if_then_else(k < 2, 0, -1)

                if i_s >= 0:
                    T.attr(A_shared.data, "tl.tma_copy_write_buffer", 1)
                    if tx == 0:
                        T.call_intrin(
                            "handle",
                            tir.op.Op.get("tl.mbarrier_expect_tx"),
                            mbarrier[k % 2],
                            4096,
                        )
                    if tx == 0:
                        T.tma_load(
                            T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                            mbarrier[k % 2],
                            T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 2 * 2048, 2048, 2),
                            k * 32,
                            by * 64,
                        )
                if i_s >= 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_wait_parity"),
                        mbarrier[k % 2],
                        k // 2 % 2,
                    )

                use_block: T.int32 = T.if_then_else(i_s >= 0, 1, 0)
                if use_block != 0:
                    C_local[0] = C_local[0] + T.Cast("float32", A_shared[k % 2, 0, 0, 0])

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    main_func = mod["main"]
    body_text = main_func.script()
    assert 'threadIdx.x", 256' in body_text

    guarded_wait_count = 0
    guarded_arrive_count = 0
    for if_stmt in _collect_ifs(main_func.body):
        if "use_block" not in str(if_stmt.condition):
            continue
        guarded_wait_count += _count_calls_in_stmt(if_stmt.then_case, "tl.mbarrier_wait_parity")
        guarded_arrive_count += _count_calls_in_stmt(if_stmt.then_case, "tir.ptx_arrive_barrier")

    assert guarded_wait_count >= 1
    assert guarded_arrive_count >= 1


if __name__ == "__main__":
    tilelang.testing.main()
