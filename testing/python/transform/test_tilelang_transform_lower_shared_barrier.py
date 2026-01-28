import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def matmul(M, N, K, block_M, block_N, block_K, mbars, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            barriers = T.alloc_barrier(mbars)  # noqa: F841
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_lower_shared_barrier():
    mbars = (1, 1, 128, 128)  # list is unhashable so we use tuple here
    kernel = matmul(1024, 1024, 1024, 128, 128, 32, mbars=mbars)

    assert f"uint64_t barriers_mem[{len(mbars)}];" in kernel.get_kernel_source()
    assert "if (tl::tl_shuffle_elect<0>()) {" in kernel.get_kernel_source()
    for i in range(len(mbars)):
        assert f"barriers[{i}].init({mbars[i]});" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
