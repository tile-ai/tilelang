import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
import torch


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    num_stages = 0

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_local = T.alloc_local((block_M, block_K), dtype)
            B_local = T.alloc_local((block_K, block_N), dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            T.clear(C_local)

            # Apply layout optimizations or define your own layout
            # (Optional).
            # T.annotate_layout(
            #     {
            #         A_local: make_swizzle_layout(A_local),
            #         B_local: make_swizzle_layout(B_local),
            #     }
            # )

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_local)

                # Or Copy with Parallel
                for k, j in T.Parallel(block_K, block_N):
                    B_local[k, j] = B[ko * block_K + k, by * block_N + j]

                for i, j, k in T.grid(block_M, block_N, block_K):
                    C_local[i, j] += A_local[i, k] * B_local[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul


def assert_matmul_codegen(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32):
    func = matmul(M, N, K, block_M, block_N, block_K)

    with tvm.target.Target("c"):
        artifact = tilelang.lower(func)

    code = artifact.kernel_source

    assert code is not None, "Code generation failed"


def test_matmul_codegen():
    assert_matmul_codegen(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)


def test_matmul_compile():
    def matmul_jit_test(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
        # a simple kernel just for jit test
        @T.prim_func
        def matmul(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
                A_local = T.alloc_local((block_M, block_K), dtype)
                B_local = T.alloc_local((block_K, block_N), dtype)
                C_local = T.alloc_local((block_M, block_N), accum_dtype)

                for p in T.serial(block_M):
                    for w in T.serial(block_N):
                        C_local[p, w] = 0
                for ko in T.serial(K // block_K):
                    for i in T.serial(block_M):
                        for k in T.serial(block_K):
                            A_local[i, k] = A[by * block_M + i, ko * block_K + k]

                    for k in T.serial(block_K):
                        for j in T.serial(block_N):
                            B_local[k, j] = B[ko * block_K + k, bx * block_N + j]

                    for i in T.serial(block_M):
                        for j in T.serial(block_N):
                            for k in T.serial(block_K):
                                C_local[i, j] += A_local[i, k] * B_local[k, j]

                for i in T.serial(block_M):
                    for j in T.serial(block_N):
                        C[by * block_M + i, bx * block_N + j] = C_local[i, j]

        return matmul

    M, N, K = 1024, 512, 512
    block_M, block_N, block_K = M // 4, N // 4, K // 4
    cpu_func = matmul_jit_test(M, N, K, block_M, block_N, block_K)
    with tvm.target.Target("c"):
        complied_fun = tilelang.compile(cpu_func, -1, execution_backend="cython")

    in_dtype = T.float16
    A = torch.randn(M, K, dtype=torch.__getattribute__(in_dtype))
    B = torch.randn(K, N, dtype=torch.__getattribute__(in_dtype))

    C = complied_fun(A, B)
    C_torch = torch.matmul(A, B)

    tilelang.testing.torch_assert_close(C, C_torch, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


if __name__ == "__main__":
    tilelang.testing.main()
