import tilelang
import tilelang.testing
import tilelang.language as T
import pytest
import torch


def shared_any_of(size, threads, scope=None):
    @T.prim_func
    def main(
        source: T.Tensor((size,), "int32"),
        result: T.Tensor((threads,), "bool"),
    ):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared((size,), "int32")
            tx = T.get_thread_binding()
            T.copy(source, shared)
            T.sync_threads()
            if scope is None:
                result[tx] = T.any_of(shared)
            else:
                result[tx] = T.any_of(shared, scope=scope)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("scope", [None, "auto", "thread", "warp"])
def test_any_of_shared_scope(scope):
    size, threads = 70, 32
    kernel = tilelang.compile(shared_any_of(size, threads, scope), target="cuda", out_idx=-1)
    for true_index in (None, 0, threads - 1, size - 1):
        source = torch.zeros(size, dtype=torch.int32, device="cuda")
        if true_index is not None:
            source[true_index] = 1
        result = kernel(source)
        expected = torch.full((threads,), true_index is not None, dtype=torch.bool, device="cuda")
        torch.testing.assert_close(result, expected)

    helper = "tl::AnyWarp" if scope == "warp" else "tl::Any"
    assert helper in kernel.get_kernel_source()


@tilelang.testing.requires_rocm
def test_any_of_shared_warp_scope_rocm():
    size, threads = 134, 64
    kernel = tilelang.compile(shared_any_of(size, threads, "warp"), target="hip", out_idx=-1)
    for true_index in (None, size - 1):
        source = torch.zeros(size, dtype=torch.int32, device="cuda")
        if true_index is not None:
            source[true_index] = 1
        result = kernel(source)
        expected = torch.full((threads,), true_index is not None, dtype=torch.bool, device="cuda")
        torch.testing.assert_close(result, expected)

    assert "tl::AnyWarp" in kernel.get_kernel_source()


def test_any_of_rejects_invalid_scope():
    with pytest.raises(ValueError, match="scope must be 'auto', 'thread' or 'warp'"):

        @T.prim_func
        def main(source: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1):
                T.evaluate(T.any_of(source, scope="block"))


def ref_program(A, B, BlockMask, block_M, block_N, block_K):
    M, K = A.shape
    N = B.shape[1]
    ref_c = torch.zeros((M, N), dtype=torch.float16, device=A.device)
    for i in range(M // block_M):
        for j in range(N // block_N):
            accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
            for k in range(K // block_K):
                if torch.any(BlockMask[i, j, k]):
                    accu += A[i * block_M : (i + 1) * block_M, k * block_K : (k + 1) * block_K].to(torch.float32) @ B[
                        k * block_K : (k + 1) * block_K, j * block_N : (j + 1) * block_N
                    ].to(torch.float32)
            ref_c[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = accu.to(torch.float16)
    return ref_c


def blocksparse_matmul_global(
    M,
    N,
    K,
    condition_dim,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    block_mask_shape = (M // block_M, N // block_N, K // block_K, condition_dim)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if T.any_of(BlockMask[by, bx, k, :]):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def blocksparse_matmul_shared(
    M,
    N,
    K,
    condition_dim,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    block_mask_shape = (M // block_M, N // block_N, K // block_K, condition_dim)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            block_mask_shared = T.alloc_shared(condition_dim, "bool")
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                for i in T.serial(condition_dim):
                    block_mask_shared[i] = BlockMask[by, bx, k, i]
                # or T.any_of(block_mask_local[0:condition_dim])
                # or T.any_of(block_mask_local[:])
                if T.any_of(block_mask_shared):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def blocksparse_matmul_local(
    M,
    N,
    K,
    condition_dim,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    block_mask_shape = (M // block_M, N // block_N, K // block_K, condition_dim)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            block_mask_local = T.alloc_local(condition_dim, "bool")
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                for i in T.serial(condition_dim):
                    block_mask_local[i] = BlockMask[by, bx, k, i]
                # or T.any_of(block_mask_local[0:condition_dim])
                # or T.any_of(block_mask_local[:])
                if T.any_of(block_mask_local):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def run_block_sparse_matmul_global(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2):
    block_M = 128
    block_N = 128
    block_K = 32
    num_stages = 2
    thread_num = 128
    enable_rasteration = True

    # Initialize input matrices A and B on the GPU with half precision
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    func = blocksparse_matmul_global(
        M,
        N,
        K,
        condition_dim,
        block_M,
        block_N,
        block_K,
        num_stages,
        thread_num,
        enable_rasteration,
    )
    kernel = tilelang.compile(func, out_idx=-1)
    # Create block mask with desired sparsity
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity
    block_mask = block_mask.view(mask_shape + (1,)).repeat(1, 1, 1, condition_dim)
    # random set the last dimension to be False
    block_mask[:, :, :, 0] = False

    # Run the compiled kernel (either tuned or default) with the inputs
    c = kernel(a, b, block_mask)

    # Compute the reference result using the naive PyTorch implementation
    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def run_block_sparse_matmul_shared(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2):
    block_M = 128
    block_N = 128
    block_K = 32
    num_stages = 2
    thread_num = 128
    enable_rasteration = True

    # Initialize input matrices A and B on the GPU with half precision
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    func = blocksparse_matmul_shared(
        M,
        N,
        K,
        condition_dim,
        block_M,
        block_N,
        block_K,
        num_stages,
        thread_num,
        enable_rasteration,
    )
    kernel = tilelang.compile(
        func,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    # Create block mask with desired sparsity
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity
    block_mask = block_mask.view(mask_shape + (1,)).repeat(1, 1, 1, condition_dim)
    # random set the last dimension to be False
    block_mask[:, :, :, 0] = False

    # Run the compiled kernel (either tuned or default) with the inputs
    c = kernel(a, b, block_mask)

    # Compute the reference result using the naive PyTorch implementation
    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def run_block_sparse_matmul_local(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2):
    block_M = 128
    block_N = 128
    block_K = 32
    num_stages = 2
    thread_num = 128
    enable_rasteration = True

    # Initialize input matrices A and B on the GPU with half precision
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    func = blocksparse_matmul_local(
        M,
        N,
        K,
        condition_dim,
        block_M,
        block_N,
        block_K,
        num_stages,
        thread_num,
        enable_rasteration,
    )
    kernel = tilelang.compile(
        func,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    # Create block mask with desired sparsity
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity
    block_mask = block_mask.view(mask_shape + (1,)).repeat(1, 1, 1, condition_dim)
    # random set the last dimension to be False
    block_mask[:, :, :, 0] = False

    # Run the compiled kernel (either tuned or default) with the inputs
    c = kernel(a, b, block_mask)

    # Compute the reference result using the naive PyTorch implementation
    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def test_block_sparse_matmul_global():
    run_block_sparse_matmul_global(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2)


def test_block_sparse_matmul_shared():
    run_block_sparse_matmul_shared(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2)


def test_block_sparse_matmul_local():
    run_block_sparse_matmul_local(M=1024, N=1024, K=1024, sparsity=0.5, condition_dim=2)


if __name__ == "__main__":
    tilelang.testing.main()
