# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import torch


def cumsum_smem_test(M, N, block_M, block_N, dim=0, reverse=False, dtype="float16"):
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


def cumsum_fragment_test(M, N, block_M, block_N, dim=0, reverse=False, dtype="float16"):
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


def run_cumsum(M, N, block_M, block_N, dim=0, reverse=False, dtype="float16", scope="smem"):
    if scope == "smem":
        program = cumsum_smem_test(M, N, block_M, block_N, dim, reverse, dtype)
    elif scope == "fragment":
        program = cumsum_fragment_test(M, N, block_M, block_N, dim, reverse, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Randn)

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

    profiler.assert_allclose(ref_program)


def test_cumsum_smem():
    # Test different sizes
    run_cumsum(1024, 1024, 128, 128)
    run_cumsum(1024, 1024, 128, 128, dim=1)
    run_cumsum(1024, 1024, 128, 128, dim=1, reverse=True)

    # Test different dtypes
    run_cumsum(256, 256, 128, 128, dtype="float32")
    run_cumsum(256, 256, 128, 128, dtype="float16")


def test_cumsum_fragment():
    run_cumsum(1024, 1024, 128, 128, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, scope="fragment")
    run_cumsum(1024, 1024, 128, 128, dim=1, reverse=True, scope="fragment")

    # Test different dtypes
    run_cumsum(256, 256, 128, 128, dtype="float32", scope="fragment")
    run_cumsum(256, 256, 128, 128, dtype="float16", scope="fragment")


if __name__ == "__main__":
    tilelang.testing.main()
