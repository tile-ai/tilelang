# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tilelang import tvm as tvm
import tilelang.testing
import torch


def elementwise_add(
    M,
    N,
    block_M,
    block_N,
    in_dtype,
    out_dtype,
    threads,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((M, N), in_dtype),
            B: T.Buffer((M, N), in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return main


def run_elementwise_add(
    M,
    N,
    in_dtype,
    out_dtype,
    block_M,
    block_N,
    num_threads=128,
):
    program = elementwise_add(
        M,
        N,
        block_M,
        block_N,
        in_dtype,
        out_dtype,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        C = torch.add(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_elementwise_add_f32():
    run_elementwise_add(
        512,
        1024,
        "float32",
        "float32",
        128,
        256,
    )


def test_elementwise_add_f16():
    run_elementwise_add(
        512,
        1024,
        "float16",
        "float16",
        128,
        256,
    )


def test_elementwise_add_i32():
    run_elementwise_add(
        512,
        1024,
        "int32",
        "int32",
        128,
        256,
    )


def test_elementwise_add_f32f16():
    run_elementwise_add(
        512,
        1024,
        "float32",
        "float16",
        128,
        256,
    )


if __name__ == "__main__":
    tilelang.testing.main()
