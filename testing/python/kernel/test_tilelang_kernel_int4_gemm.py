import tilelang
import tilelang.testing
import tilelang.language as T


def matmul_nt_int4(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int4),
        B: T.Tensor((N, K), T.int4),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.int4)
            B_shared = T.alloc_shared((block_N, block_K), T.int4)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(8, 0)
def test_compile_int4_gemm_tgemm():
    func = matmul_nt_int4(1024, 1024, 1024, 128, 128, 64)
    kernel = tilelang.compile(func, out_idx=-1)
    src = kernel.get_kernel_source()
    assert src is not None
    assert "s4.s4.s32" in src or "int4" in src


if __name__ == "__main__":
    tilelang.testing.main()
