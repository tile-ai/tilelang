#!/usr/bin/env python3

import tilelang
import tilelang.language as T
import torch

def test_specific_function():
    """Test with only the needed function defined"""

    # Only define the function we need for ieee_add
    @T.prim_func
    def main_two_arg(
            A: T.Tensor((128, 128), "float32"),
            B: T.Tensor((128, 128), "float32"),
            C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32), threads=128) as (bx, by):
            for i, j in T.Parallel(32, 32):
                C[by * 32 + i, bx * 32 + j] = T.ieee_add(
                    A[by * 32 + i, bx * 32 + j],
                    B[by * 32 + i, bx * 32 + j],
                    "rn"
                )

    print("Compiling main_two_arg...")

    # Test compilation
    kernel = tilelang.compile(
        main_two_arg,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        }
    )

    print("✓ Compilation successful!")

if __name__ == "__main__":
    try:
        test_specific_function()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()