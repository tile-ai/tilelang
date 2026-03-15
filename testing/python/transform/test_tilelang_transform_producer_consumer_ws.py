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

            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.create_list_of_mbarrier"),
                128,
                128,
                128,
                128,
                128,
                128,
            )

            for k in T.serial(16, annotations={"num_stages": T.int32(3)}):
                if v == 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_expect_tx"),
                        T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3),
                        4096,
                    )
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                        T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3),
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        by * 64,
                    )
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.mbarrier_wait_parity"),
                    T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3),
                    k // 3 % 2,
                )

                if v == 0:
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.mbarrier_expect_tx"),
                        T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3 + 3),
                        4096,
                    )
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, B.data, 512, 512, 2, 1024, 64, 32, 1, 1, 0, 3, 2, 0),
                        T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3 + 3),
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        bx * 64,
                    )
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.mbarrier_wait_parity"),
                    T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), k % 3 + 3),
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
    create_list_calls = _collect_calls(main_func.body, "tl.create_list_of_mbarrier")
    assert len(create_list_calls) == 1
    assert len(create_list_calls[0].args) == 6


if __name__ == "__main__":
    tilelang.testing.main()
