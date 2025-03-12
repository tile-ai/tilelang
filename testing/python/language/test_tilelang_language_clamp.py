# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl

def clamp(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = T.clamp(A_shared[i], min=min, max=max)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_clamp(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = clamp(
        N,
        block_N,
        dtype,
        min,
        max
    )

    mod, params = tl.lower(program)
    profiler = tl.Profiler(mod, params, [1], tl.TensorSupplyType.Integer)

    def ref_program(A):
        import torch

        output = torch.clamp(A, min, max)
        return output

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_clamp():
    # GEMM tests for float16
    run_clamp(1024, 128, "float16", -0.05, 0.05)
    run_clamp(1024, 64,  "float16", -0.10, None)
    run_clamp(1024, 64,  "float16", None,  0.10)
    run_clamp(1024, 128, "float32", -0.06, 0.05)
    run_clamp(1024, 128, "float32", -0.05, None)
    run_clamp(1024, 128, "float32", None,  0.05)

if __name__ == "__main__":
    tilelang.testing.main()
