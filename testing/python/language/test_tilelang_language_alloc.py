# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang.testing
from tilelang.utils.tensor import map_torch_type

def alloc_variable(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_variable(dtype)
            tmp = 1
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_variable(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = alloc_variable(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    profiler = kernel.get_profiler()
    code = kernel.get_kernel_source()
    assert "tmp =" in code

def test_alloc_variable():
    run_alloc_variable(1024, 128, "float16")

def alloc_variable_add(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_variable(dtype)
            tmp = 1
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = A_shared[i] + tmp
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_variable_add(
    N,
    block_N,
    dtype,
):
    program = alloc_variable_add(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    profiler = kernel.get_profiler()
    code = kernel.get_kernel_source()
    assert "tmp =" in code

def test_alloc_variable_add():
    run_alloc_variable_add(1024, 128, "float16")

if __name__ == "__main__":
    tilelang.testing.main()
