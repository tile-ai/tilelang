import tilelang
import tilelang.language as T
import torch


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy_mask_parallel(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            tx = T.get_thread_binding(0)

            if tx < 128:
                for i, k in T.Parallel(block_M, block_N):
                    A_shared[i, k] = A[by * block_M + i, bx * block_N + k]

            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def run_tilelang_copy_mask_parallel(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy_mask_parallel(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program, out_idx=[1], target="cuda", pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_mask_parallel():
    run_tilelang_copy_mask_parallel(M=1024, N=1024, block_M=128, block_N=128)


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy_mask_copy(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            tx = T.get_thread_binding(0)

            if tx < 128:
                T.copy(A[by * block_M, bx * block_N], A_shared)

            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def run_tilelang_copy_mask_copy(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy_mask_copy(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program, out_idx=[1], target="cuda", pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_mask_copy():
    run_tilelang_copy_mask_copy(M=1024, N=1024, block_M=128, block_N=128)


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy_mask_parallel_range(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            tx = T.get_thread_binding(0)

            if tx >= 128 and tx < 256:
                for i, k in T.Parallel(block_M, block_N):
                    A_shared[i, k] = A[by * block_M + i, bx * block_N + k]

            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def run_tilelang_copy_mask_parallel_range(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy_mask_parallel_range(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program, out_idx=[1], target="cuda", pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_mask_parallel_range():
    run_tilelang_copy_mask_parallel_range(M=1024, N=1024, block_M=128, block_N=128)


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy_mask_copy_range(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            tx = T.get_thread_binding(0)

            if tx >= 128 and tx < 256:
                T.copy(A[by * block_M, bx * block_N], A_shared)

            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def run_tilelang_copy_mask_copy_range(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy_mask_copy_range(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program, out_idx=[1], target="cuda", pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_mask_copy_range():
    run_tilelang_copy_mask_copy_range(M=1024, N=1024, block_M=128, block_N=128)


if __name__ == "__main__":
    test_tilelang_copy_mask_copy_range()
