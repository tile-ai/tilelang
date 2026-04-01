import re

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.layout import make_cutlass_metadata_layout


def _compile_tvm_ffi(func, pass_configs=None):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs or {},
        )
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_keeps_full_producer_extent_for_lowered_simt_copy():
    M, N, K = 128, 64, 64
    block_M, block_N, block_K = 64, 64, 32
    num_stages = 2
    threads = 256
    e_factor = 8

    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), "uint8"),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), "uint8")
            C_frag = T.alloc_fragment((block_M, block_N), T.float32)
            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, arch="9.0", block_k=block_K),
                    E_shared: make_cutlass_metadata_layout(
                        E_shared,
                        mma_dtype=T.float16,
                        arch="9.0",
                        block_k=block_K,
                    ),
                }
            )
            T.disable_warp_group_reg_alloc()
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, False, False)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    kernel = _compile_tvm_ffi(main)
    src = kernel.get_kernel_source()
    flat_src = " ".join(src.split())

    assert "__launch_bounds__(512, 1)" in src
    assert "if (256 <= ((int)threadIdx.x)) {" in flat_src
    assert "tl::tl_shuffle_elect<256>()" in src or "if (((int)threadIdx.x) == 256) {" in src
    assert re.search(r"tl::__sync_thread_partial<\d+, 256>\(\);", src), src


if __name__ == "__main__":
    tilelang.testing.main()
