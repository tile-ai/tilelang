from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import torch
import tilelang.language as T


def cumsum_smem_test(M, N, block_M, block_N, dim=0, reverse=False, dtype=T.float32):
    @T.prim_func
    def cumsum(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.cumsum(src=A_shared, dim=dim, reverse=reverse)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return cumsum


def cumsum_fragment_test(M, N, block_M, block_N, dim=0, reverse=False, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def cumsum(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            A_fragment = T.alloc_fragment((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, A_fragment)
            T.cumsum(src=A_fragment, dim=dim, reverse=reverse)
            T.copy(A_fragment, B[by * block_M, bx * block_N])

    return cumsum


def run_cumsum(M, N, block_M, block_N, dim=0, reverse=False, dtype=T.float32, scope="smem"):
    if scope == "smem":
        program = cumsum_smem_test(M, N, block_M, block_N, dim, reverse, dtype)
    elif scope == "fragment":
        program = cumsum_fragment_test(M, N, block_M, block_N, dim, reverse, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)

    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()

    def ref_program(A):
        ref_b = torch.empty_like(A)
        for i in range(M // block_M):
            for j in range(N // block_N):
                ref_b[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = A[
                    i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N
                ].cumsum(dim=dim)
                if reverse:
                    ref_b[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = (
                        A[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N]
                        .flip(dims=[dim])
                        .cumsum(dim=dim)
                        .flip(dims=[dim])
                    )
        return ref_b

    tilelang_res = jit_kernel(A)
    ref_res = ref_program(A)
    torch.testing.assert_close(tilelang_res, ref_res, atol=1e-3, rtol=1e-3)


def cumsum_smem_test_1d(N, block_N, reverse=False, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def cumsum(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared((block_N,), dtype)

            T.copy(A[bx * block_N], A_shared)
            T.cumsum(src=A_shared, dim=0, reverse=reverse)
            T.copy(A_shared, B[bx * block_N])

    return cumsum


def cumsum_fragment_test_1d(N, block_N, reverse=False, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def cumsum(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared((block_N,), dtype)
            A_fragment = T.alloc_fragment((block_N,), dtype)

            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, A_fragment)
            T.cumsum(src=A_fragment, dim=0, reverse=reverse)
            T.copy(A_fragment, B[bx * block_N])

    return cumsum


def run_cumsum_1d(N, block_N, reverse=False, dtype=T.float32, scope="smem"):
    if scope == "smem":
        program = cumsum_smem_test_1d(N, block_N, reverse, dtype)
    elif scope == "fragment":
        program = cumsum_fragment_test_1d(N, block_N, reverse, dtype)
    else:
        raise ValueError(f"Unknown scope {scope}")

    jit_kernel = tl.compile(program, out_idx=-1)
    A = torch.randn(N, dtype=getattr(torch, dtype)).cuda()

    def ref_program(A):
        ref_b = torch.empty_like(A)
        num_blocks = (N + block_N - 1) // block_N
        for j in range(num_blocks):
            start = j * block_N
            end = min(start + block_N, N)
            chunk = A[start:end]
            if reverse:
                chunk = torch.flip(chunk, dims=[0])
            chunk = chunk.cumsum(dim=0)
            if reverse:
                chunk = torch.flip(chunk, dims=[0])
            ref_b[start:end] = chunk
        return ref_b

    tilelang_res = jit_kernel(A)
    ref_res = ref_program(A)
    torch.testing.assert_close(tilelang_res, ref_res, atol=1e-3, rtol=1e-3)


def test_cumsum_smem():
    # Test different sizes
    run_cumsum(1024, 1024, 128, 128)
    run_cumsum(1024, 1024, 128, 128, dim=1)
    run_cumsum(1024, 1024, 128, 128, dim=1, reverse=True)

    # Test different dtypes
    run_cumsum(256, 256, 128, 128, dtype=T.float32)
    run_cumsum(256, 256, 128, 128, dtype=T.float32)


def test_cumsum_fragment():
    run_cumsum(1024, 1024, 128, 128, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, reverse=True, scope="fragment")

    # Test different dtypes
    run_cumsum(256, 256, 128, 128, dtype=T.float32, scope="fragment")
    run_cumsum(256, 256, 128, 128, dtype=T.float32, scope="fragment")


def test_cumsum_smem_1d():
    run_cumsum_1d(1024, 128)
    run_cumsum_1d(1024, 128, reverse=True)


def test_cumsum_fragment_1d():
    run_cumsum_1d(1024, 128, scope="fragment")
    run_cumsum_1d(1024, 128, reverse=True, scope="fragment")


def cumsum_region_test_1d(N, chunk_size, reverse=False, dtype=T.float32):
    """Test cumsum with buffer region (slice) as input."""
    import tilelang.language as T

    @T.prim_func
    def cumsum_region(
        InputG_fragment: T.Tensor((N,), dtype),
        OutputG_fragment: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, chunk_size), threads=chunk_size) as bx:
            i = bx
            chunk_start = i * chunk_size
            # Copy region to shared memory first (cumsum only supports shared memory)
            A_shared = T.alloc_shared((chunk_size,), dtype)
            T.copy(InputG_fragment[chunk_start : chunk_start + chunk_size], A_shared)
            # Test cumsum with region input - in-place operation on shared memory
            # This demonstrates the feature: T.cumsum(region, dim=0)
            T.cumsum(src=A_shared, dim=0, reverse=reverse)
            # Copy result back to global memory
            T.copy(A_shared, OutputG_fragment[chunk_start : chunk_start + chunk_size])

    return cumsum_region


def run_cumsum_region_1d(N, chunk_size, reverse=False, dtype=T.float32):
    """Run test for cumsum with region input."""
    program = cumsum_region_test_1d(N, chunk_size, reverse, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    A = torch.randn(N, dtype=getattr(torch, dtype)).cuda()

    def ref_program(A):
        ref_b = torch.empty_like(A)
        num_blocks = (N + chunk_size - 1) // chunk_size
        for j in range(num_blocks):
            start = j * chunk_size
            end = min(start + chunk_size, N)
            chunk = A[start:end].clone()
            if reverse:
                chunk = torch.flip(chunk, dims=[0])
            chunk = chunk.cumsum(dim=0)
            if reverse:
                chunk = torch.flip(chunk, dims=[0])
            ref_b[start:end] = chunk
        return ref_b

    tilelang_res = jit_kernel(A)
    ref_res = ref_program(A)
    torch.testing.assert_close(tilelang_res, ref_res, atol=1e-3, rtol=1e-3)


def cumsum_region_test_2d(M, N, block_M, block_N, dim=0, reverse=False, dtype=T.float32):
    """Test cumsum with buffer region (slice) as input in 2D."""
    import tilelang.language as T

    @T.prim_func
    def cumsum_region(
        InputG_fragment: T.Tensor((M, N), dtype),
        OutputG_fragment: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            chunk_start_M = by * block_M
            chunk_start_N = bx * block_N
            # Copy region to shared memory first (cumsum only supports shared memory)
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(
                InputG_fragment[chunk_start_M : chunk_start_M + block_M, chunk_start_N : chunk_start_N + block_N],
                A_shared,
            )
            # Test cumsum with 2D region input - in-place operation on shared memory
            T.cumsum(src=A_shared, dim=dim, reverse=reverse)
            # Copy result back to global memory
            T.copy(
                A_shared,
                OutputG_fragment[chunk_start_M : chunk_start_M + block_M, chunk_start_N : chunk_start_N + block_N],
            )

    return cumsum_region


def run_cumsum_region_2d(M, N, block_M, block_N, dim=0, reverse=False, dtype=T.float32):
    """Run test for cumsum with 2D region input."""
    program = cumsum_region_test_2d(M, N, block_M, block_N, dim, reverse, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()

    def ref_program(A):
        ref_b = torch.empty_like(A)
        num_blocks_M = (M + block_M - 1) // block_M
        num_blocks_N = (N + block_N - 1) // block_N
        for i in range(num_blocks_M):
            for j in range(num_blocks_N):
                start_M = i * block_M
                end_M = min(start_M + block_M, M)
                start_N = j * block_N
                end_N = min(start_N + block_N, N)
                chunk = A[start_M:end_M, start_N:end_N].clone()
                if reverse:
                    chunk = torch.flip(chunk, dims=[dim])
                chunk = chunk.cumsum(dim=dim)
                if reverse:
                    chunk = torch.flip(chunk, dims=[dim])
                ref_b[start_M:end_M, start_N:end_N] = chunk
        return ref_b

    tilelang_res = jit_kernel(A)
    ref_res = ref_program(A)
    torch.testing.assert_close(tilelang_res, ref_res, atol=1e-3, rtol=1e-3)


def test_cumsum_region_1d():
    """Test cumsum with 1D region input."""
    # Test normal cumsum with region input
    run_cumsum_region_1d(1024, 128)
    # Test reverse cumsum with region input
    run_cumsum_region_1d(1024, 128, reverse=True)
    # Test with different chunk sizes
    run_cumsum_region_1d(512, 64)
    run_cumsum_region_1d(2048, 256)
    # Tail coverage (non-divisible size)
    run_cumsum_region_1d(1000, 128)


def test_cumsum_region_2d():
    """Test cumsum with 2D region input."""
    # Test 2D cumsum along dim 0
    run_cumsum_region_2d(1024, 1024, 128, 128, dim=0)
    # Test 2D cumsum along dim 1
    run_cumsum_region_2d(1024, 1024, 128, 128, dim=1)
    # Test reverse cumsum
    run_cumsum_region_2d(512, 512, 64, 64, dim=1, reverse=True)
    # Tail coverage (non-divisible size)
    run_cumsum_region_2d(1000, 1000, 128, 128, dim=1)


if __name__ == "__main__":
    tilelang.testing.main()
