from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    order,
    stage,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    order,
    stage,
):
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    trans_A = False
    trans_B = False
    in_dtype = T.float16
    out_dtype = T.float16
    dtypeAccum = T.float32
    num_threads = 128
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        order,
        stage,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype == T.float32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_pipeline_order_stage():
    run_gemm(order=[0, 1, 2], stage=[0, 0, 1])
    run_gemm(order=[0, 1, 2], stage=[0, 0, 2])
    run_gemm(order=[1, 2, 0], stage=[0, 0, 2])
    run_gemm(order=[1, 2, 0], stage=[0, 0, 1])


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    block_mask_shape = (M // block_M, N // block_N, K // block_K)

    import tilelang.language as T

    @T.prim_func
    def block_sparse_matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            block_mask = T.alloc_local((1,), "bool")
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                block_mask[0] = BlockMask[by, bx, k]
                if block_mask[0]:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return block_sparse_matmul


def run_blocksparse_matmul(num_stages):
    import torch

    M = 256
    N = 256
    K = 256
    block_M = 128
    block_N = 128
    block_K = 32
    sparsity = 0.5

    # Initialize input matrices A and B on the GPU with half precision
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    kernel = blocksparse_matmul(M, N, K, block_M=block_M, block_N=block_N, block_K=block_K, num_stages=num_stages)
    print(kernel.get_kernel_source())
    # Create block mask with desired sparsity
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity

    # Run the compiled kernel (either tuned or default) with the inputs
    c = kernel(a, b, block_mask)

    def ref_program(A, B, BlockMask, block_M, block_N, block_K):
        ref_c = torch.zeros((M, N), dtype=torch.float16, device=A.device)
        for i in range(M // block_M):
            for j in range(N // block_N):
                accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
                for k in range(K // block_K):
                    if BlockMask[i, j, k]:
                        accu += A[i * block_M : (i + 1) * block_M, k * block_K : (k + 1) * block_K].to(torch.float32) @ B[
                            k * block_K : (k + 1) * block_K, j * block_N : (j + 1) * block_N
                        ].to(torch.float32)
                ref_c[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = accu.to(torch.float16)
        return ref_c

    # Compute the reference result using the naive PyTorch implementation
    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def test_blocksparse_matmul():
    run_blocksparse_matmul(num_stages=1)
    run_blocksparse_matmul(num_stages=2)
    run_blocksparse_matmul(num_stages=3)


if __name__ == "__main__":
    tilelang.testing.main()
