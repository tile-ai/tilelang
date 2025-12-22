from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm import tir


def test_inject_set_max_nreg():
    """Test the InjectSetMaxNReg pass"""

    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16), B: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        v = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[by * 64, 0:512], B[0:512, bx * 64])
            T.writes()

            # Add set_max_nreg hints
            T.annotate_producer_reg_dealloc(24)  # Producer: decrease to 24
            T.annotate_consumer_reg_alloc(240)  # Consumer: increase to 240

            A_shared = T.alloc_buffer((3, 1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.alloc_buffer((3, 1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((32,), scope="local")

            T.create_list_of_mbarrier(128, 128, 128, 128, 128, 128)
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                # Producer branch - should have set_max_nreg(24, 0)
                for k in range(16):
                    T.mbarrier_wait_parity(T.get_mbarrier(k % 3 + 3), T.bitwise_xor(k // 3 % 2, 1))
                    if v - 128 == 0:
                        T.tma_load(
                            T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                            T.get_mbarrier(k % 3),
                            T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                            k * 32,
                            by * 64,
                        )
                    T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [T.get_mbarrier(k % 3)]))
            else:
                # Consumer branch - should have set_max_nreg(240, 1)
                for k in range(16):
                    T.mbarrier_wait_parity(T.get_mbarrier(k % 3), k // 3 % 2)
                    T.call_extern(
                        "handle",
                        "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 1),
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 1),
                        T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
                    )
                    T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [T.get_mbarrier(k % 3 + 3)]))

    # Apply the InjectSetMaxNReg pass
    func = before
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    # Check that set_max_nreg calls are properly injected
    main_func = mod["main"]
    set_max_nreg_calls = []

    def collect_set_max_nreg(stmt):
        if (
            isinstance(stmt, tvm.tir.Evaluate)
            and hasattr(stmt.value, "op")
            and hasattr(stmt.value.op, "name")
            and stmt.value.op.name == "tl.set_max_nreg"
        ):
            set_max_nreg_calls.append(stmt.value)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_set_max_nreg)

    # We should have at least 2 set_max_nreg calls (one for producer, one for consumer)
    assert len(set_max_nreg_calls) >= 2, f"Expected at least 2 set_max_nreg calls, got {len(set_max_nreg_calls)}"

    print("InjectSetMaxNReg test passed!")


def test_inject_set_max_nreg_no_set_max_nreg():
    """Test the InjectSetMaxNReg pass with no_set_max_nreg"""

    @T.prim_func
    def before_no_set_max_nreg(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        v = T.launch_thread("threadIdx.x", 128)

        with T.block(""):
            T.reads(A[bx * 64, 0:64])
            T.writes()

            # Add no_set_max_nreg to disable register hinting
            T.disable_warp_group_reg_alloc()

            T.create_list_of_mbarrier(128, 128)
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                # Producer branch - should not have set_max_nreg calls
                T.evaluate(0)
            else:
                # Consumer branch - should not have set_max_nreg calls
                T.evaluate(0)

    # Apply the InjectSetMaxNReg pass
    func = before_no_set_max_nreg
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    # Check that no set_max_nreg calls are injected when no_set_max_nreg is present
    main_func = mod["main"]
    set_max_nreg_calls = []

    def collect_set_max_nreg(stmt):
        if (
            isinstance(stmt, tvm.tir.Evaluate)
            and hasattr(stmt.value, "op")
            and hasattr(stmt.value.op, "name")
            and stmt.value.op.name == "tl.set_max_nreg"
        ):
            set_max_nreg_calls.append(stmt.value)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_set_max_nreg)

    # Should have no set_max_nreg calls when no_set_max_nreg is present
    assert len(set_max_nreg_calls) == 0, f"Expected 0 set_max_nreg calls when no_set_max_nreg is present, got {len(set_max_nreg_calls)}"

    print("InjectSetMaxNReg with no_set_max_nreg test passed!")


if __name__ == "__main__":
    # tilelang.testing.main()
    test_inject_set_max_nreg()
