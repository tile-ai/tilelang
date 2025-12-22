import tilelang
import tilelang.language as T


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            X_shared = A_shared[:block_M, :block_K]
            X_local = C_local[:block_M, :block_K]
            T.clear(X_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                aliased_offset = T.int32()
                T.let(aliased_offset, ko * block_K)
                T.copy(A[by * block_M, aliased_offset], X_shared)

                # Demonstrate parallelized copy from global to shared for B
                T.copy(B[bx * block_N, ko * block_K], B_shared[:block_N, :block_K])

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(X_shared, B_shared, C_local, transpose_B=True)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    program = matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    kernel = tilelang.compile(program, out_idx=[2], target="cuda")
    kernel.run_once()


def test_matmul():
    run_matmul(1024, 1024, 1024, 128, 128, 32)


if __name__ == "__main__":
    test_matmul()
