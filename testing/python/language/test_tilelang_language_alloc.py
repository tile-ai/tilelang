# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang.testing


def alloc_var(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_var(dtype)
            tmp = 1  # noqa: F841
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_var(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = alloc_var(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source()
    assert "tmp =" in code


def test_alloc_var():
    run_alloc_var(1024, 128, "float16")


def alloc_var_add(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_var(dtype)
            tmp = 1  # noqa: F841
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = A_shared[i] + tmp
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_var_add(
    N,
    block_N,
    dtype,
):
    program = alloc_var_add(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source()
    assert "tmp =" in code


def test_alloc_var_add():
    run_alloc_var_add(1024, 128, "float16")


if __name__ == "__main__":
    tilelang.testing.main()
