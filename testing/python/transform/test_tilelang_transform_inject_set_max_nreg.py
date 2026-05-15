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

            mbars = T.alloc_barrier([128, 128, 128, 128, 128, 128])
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                # Producer branch - should have set_max_nreg(24, 0)
                for k in range(16):
                    T.mbarrier_wait_parity(mbars[k % 3 + 3], T.bitwise_xor(k // 3 % 2, 1))
                    if v - 128 == 0:
                        T.tma_load(
                            T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                            mbars[k % 3],
                            T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                            k * 32,
                            by * 64,
                        )
                    T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [mbars[k % 3]]))
            else:
                # Consumer branch - should have set_max_nreg(240, 1)
                for k in range(16):
                    T.mbarrier_wait_parity(mbars[k % 3], k // 3 % 2)
                    T.call_extern(
                        "handle",
                        "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 1),
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 1),
                        T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
                    )
                    T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [mbars[k % 3 + 3]]))

    # Apply the InjectSetMaxNReg pass
    func = before
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    script = mod.script()
    producer_branch = script.index("if v_2 >= 128:")
    consumer_branch = script.index("else:", producer_branch)
    reg_dealloc = script.index("T.set_max_nreg(24, 0)")
    reg_alloc = script.index("T.set_max_nreg(240, 1)")

    assert producer_branch < reg_dealloc < consumer_branch
    assert consumer_branch < reg_alloc

    print("InjectSetMaxNReg test passed!")


def test_raw_set_max_nreg_keeps_legacy_behavior_with_simt_copy():
    """Raw T.set_max_nreg should stay in place instead of being treated as annotation."""

    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16), B: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        v = T.launch_thread("threadIdx.x", 256)

        with T.block(""):
            T.reads(A[bx * 64, 0:64])
            T.writes(B[bx * 64, 0:64])

            A_shared = T.alloc_buffer((128,), T.float16, scope="shared")
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                T.set_max_nreg(80, 0)
                A_shared[v - 128] = A[bx * 64, v - 128]
            else:
                T.set_max_nreg(240, 1)
                B[bx * 64, v] = A_shared[v]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    script = mod.script()
    assert script.count("T.set_max_nreg(80, 0)") == 1
    assert script.count("T.set_max_nreg(240, 1)") == 1
    assert script.count("T.set_max_nreg(") == 2


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

            mbars = T.alloc_barrier([128, 128])  # noqa: F841
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

    script = mod.script()
    assert "T.set_max_nreg(" not in script

    print("InjectSetMaxNReg with no_set_max_nreg test passed!")


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_auto_ws_reg_hints_lower_into_matching_role_scopes():
    """Producer/consumer reg hints should be emitted inside the auto-WS branches."""

    M = N = K = 256
    block_m = block_n = 128
    block_k = 32

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.annotate_producer_reg_dealloc(40)
            T.annotate_consumer_reg_alloc(232)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=3):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                T.copy(B[ko * block_k, bx * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    }
    tl.disable_cache()
    try:
        kernel_mod = tl.compile(kernel, target="cuda", pass_configs=pass_configs, out_idx=[-1])
        src = kernel_mod.get_kernel_source()
    finally:
        tl.enable_cache()

    producer_branch = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_branch = src.index("} else {", producer_branch)
    reg_dealloc = src.index("tl::warpgroup_reg_dealloc<40>();")
    reg_alloc = src.index("tl::warpgroup_reg_alloc<232>();")

    assert "warpgroup_reg_dealloc" not in src[:producer_branch]
    assert "warpgroup_reg_alloc" not in src[:producer_branch]
    assert producer_branch < reg_dealloc < consumer_branch
    assert consumer_branch < reg_alloc


if __name__ == "__main__":
    # tilelang.testing.main()
    test_inject_set_max_nreg()
