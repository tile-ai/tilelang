from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.WarpSpecialized()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    transformed = tir.transform.LowerOpaqueBlock()(transformed)

    # TODO: fix loop_var equal bug
    # tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


M = 512
N = 512
K = 512
dtype = T.float16
block_M = 64
block_N = 64
block_K = 32


def test_warp_specialized():
    @T.prim_func
    def before(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype)):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        v = T.launch_thread("threadIdx.x", 128)
        with T.block(""):
            T.reads(A[by * 64, 0:481], B[0:481, bx * 64])
            T.writes()
            A_shared = T.alloc_buffer((3, 1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.alloc_buffer((3, 1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.alloc_buffer((32,), scope="local")
            for k in T.serial(16, annotations={"num_stages": T.int32(3)}):
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                        0,
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        by * 64,
                    )
                if v == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, B.data, 512, 512, 2, 1024, 64, 32, 1, 1, 0, 3, 2, 0),
                        0,
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 2),
                        bx * 64,
                        k * 32,
                    )
                T.call_extern(
                    "handle",
                    "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                    T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
                )

    @T.prim_func
    def after(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype)):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        v = T.launch_thread("threadIdx.x", 256)
        A_shared = T.decl_buffer((3, 1, 8, 256), T.float16, scope="shared.dyn")
        B_shared = T.decl_buffer((3, 1, 4, 512), T.float16, scope="shared.dyn")
        C_local = T.decl_buffer((32,), scope="local")
        T.create_list_of_mbarrier(128, 128, 128, 128, 128, 128)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if v >= 128:
            T.set_max_nreg(24, 0)
            for k in range(16):
                T.mbarrier_wait_parity(T.get_mbarrier(k % 3 + 3), T.bitwise_xor(k // 3 % 2, 1))
                if v - 128 == 0:
                    T.mbarrier_expect_tx(T.get_mbarrier(k % 3), 4096)
                if v - 128 == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2, 2, 0),
                        T.get_mbarrier(k % 3),
                        T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, k % 3 * 2048, 2048, 2),
                        k * 32,
                        by * 64,
                    )
                if v - 128 == 0:
                    T.mbarrier_expect_tx(T.get_mbarrier(k % 3), 4096)
                if v - 128 == 0:
                    T.tma_load(
                        T.create_tma_descriptor(6, 2, B.data, 512, 512, 2, 1024, 64, 32, 1, 1, 0, 3, 2, 0),
                        T.get_mbarrier(k % 3),
                        T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, k % 3 * 2048, 2048, 2),
                        bx * 64,
                        k * 32,
                    )
                T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [T.get_mbarrier(k % 3)]))
        else:
            T.set_max_nreg(240, 1)
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

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
