#!/usr/bin/env python3

import tilelang
import tilelang.language as T
import torch
import tilelang.testing

def run_ieee_math_test_debug(mathop_name,
                       mathop_func,
                       rounding_mode="rn",
                       M=128,
                       N=128,
                       block_M=32,
                       block_N=32,
                       dtype="float32"):
    """
    Debug version of the test
    """

    print(f"=== DEBUGGING {mathop_name} ===")
    print(f"mathop_func: {mathop_func}")
    print(f"rounding_mode: {rounding_mode}")

    @T.prim_func
    def main_single_arg(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = mathop_func(
                    A[by * block_M + i, bx * block_N + j], rounding_mode
                )

    @T.prim_func
    def main_two_arg(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = mathop_func(
                    A[by * block_M + i, bx * block_N + j],
                    B[by * block_M + i, bx * block_N + j],
                    rounding_mode
                )

    @T.prim_func
    def main_fmaf(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                print(f"DEBUG: About to call {mathop_func} with 4 args")
                D[by * block_M + i, bx * block_N + j] = mathop_func(
                    A[by * block_M + i, bx * block_N + j],
                    B[by * block_M + i, bx * block_N + j],
                    C[by * block_M + i, bx * block_N + j],
                    rounding_mode
                )

    # Choose appropriate main function based on operation
    if mathop_name == "ieee_fmaf":
        main_func = main_fmaf
        out_idx = [3]
        num_inputs = 3
        print("Selected: main_fmaf")
    elif mathop_name in ["ieee_add", "ieee_sub", "ieee_mul", "ieee_fdiv"]:
        main_func = main_two_arg
        out_idx = [2]
        num_inputs = 2
        print("Selected: main_two_arg")
    else:  # Single argument operations
        main_func = main_single_arg
        out_idx = [1]
        num_inputs = 1
        print("Selected: main_single_arg")

    print(f"About to compile with function: {main_func}")

    # Test compilation
    kernel = tilelang.compile(
        main_func,
        out_idx=out_idx,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        }
    )

if __name__ == "__main__":
    try:
        run_ieee_math_test_debug("ieee_add", T.ieee_add, rounding_mode="rn")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()