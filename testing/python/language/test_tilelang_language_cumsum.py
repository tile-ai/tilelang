from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import torch


def cumsum_smem_test(M, N, block_M, block_N, dim=0, reverse=False, dtype="float32"):
    import tilelang.language as T

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


def cumsum_fragment_test(M, N, block_M, block_N, dim=0, reverse=False, dtype="float32"):
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


def run_cumsum(M, N, block_M, block_N, dim=0, reverse=False, dtype="float32", scope="smem"):
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
                ref_b[i * block_M:(i + 1) * block_M,
                      j * block_N:(j + 1) * block_N] = A[i * block_M:(i + 1) * block_M, j *
                                                         block_N:(j + 1) * block_N].cumsum(dim=dim)
                if reverse:
                    ref_b[i * block_M:(i + 1) * block_M, j * block_N:(j + 1) *
                          block_N] = A[i * block_M:(i + 1) * block_M, j * block_N:(j + 1) *
                                       block_N].flip(dims=[dim]).cumsum(dim=dim).flip(dims=[dim])
        return ref_b

    tilelang_res = jit_kernel(A)
    ref_res = ref_program(A)
    torch.testing.assert_close(tilelang_res, ref_res, atol=1e-3, rtol=1e-3)


def cumsum_smem_test_1d(N, block_N, reverse=False, dtype="float32"):
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


def cumsum_fragment_test_1d(N, block_N, reverse=False, dtype="float32"):
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


def run_cumsum_1d(N, block_N, reverse=False, dtype="float32", scope="smem"):
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
    run_cumsum(256, 256, 128, 128, dtype="float32")
    run_cumsum(256, 256, 128, 128, dtype="float32")


def test_cumsum_fragment():
    run_cumsum(1024, 1024, 128, 128, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, reverse=True, scope="fragment")

    # Test different dtypes
    run_cumsum(256, 256, 128, 128, dtype="float32", scope="fragment")
    run_cumsum(256, 256, 128, 128, dtype="float32", scope="fragment")


def test_cumsum_smem_1d():
    run_cumsum_1d(1024, 128)
    run_cumsum_1d(1024, 128, reverse=True)


def test_cumsum_fragment_1d():
    run_cumsum_1d(1024, 128, scope="fragment")
    run_cumsum_1d(1024, 128, reverse=True, scope="fragment")


if __name__ == "__main__":
    tilelang.testing.main()
