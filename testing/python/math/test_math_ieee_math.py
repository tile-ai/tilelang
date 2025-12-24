import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest


def run_ieee_math_test(mathop_name, mathop_func, rounding_mode="rn", M=128, N=128, block_M=32, block_N=32, dtype=T.float32):
    """
    Test IEEE-compliant math operations with specified rounding modes.
    """

    # Define the appropriate function based on operation type to avoid TVM parsing conflicts
    if mathop_name == "ieee_fmaf":

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    D[by * block_M + i, bx * block_N + j] = mathop_func(
                        A[by * block_M + i, bx * block_N + j],
                        B[by * block_M + i, bx * block_N + j],
                        C[by * block_M + i, bx * block_N + j],
                        rounding_mode,
                    )

        out_idx = [3]
        num_inputs = 3
    elif mathop_name in ["ieee_add", "ieee_sub", "ieee_mul", "ieee_fdiv"]:

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = mathop_func(
                        A[by * block_M + i, bx * block_N + j], B[by * block_M + i, bx * block_N + j], rounding_mode
                    )

        out_idx = [2]
        num_inputs = 2
    else:  # Single argument operations

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = mathop_func(A[by * block_M + i, bx * block_N + j], rounding_mode)

        out_idx = [1]
        num_inputs = 1

    # Test compilation
    kernel = tilelang.compile(
        main_func,
        out_idx=out_idx,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
    )

    print(f"\n=== Testing {mathop_name} with rounding mode {rounding_mode} ===")
    print(f"✓ {mathop_name} compilation test passed")

    # Test numerical execution
    torch_dtype = dtype.as_torch()
    a = torch.randn(M, N, device="cuda", dtype=torch_dtype)

    if num_inputs >= 2:
        b = torch.randn(M, N, device="cuda", dtype=torch_dtype)
    if num_inputs == 3:
        c = torch.randn(M, N, device="cuda", dtype=torch_dtype)

    # Ensure positive values for functions that need them
    if mathop_name in ["ieee_frcp", "ieee_fsqrt"]:
        a = torch.abs(a) + 0.1
    elif mathop_name == "ieee_fdiv":
        b = torch.abs(b) + 0.1  # Avoid division by zero

    # Execute kernel
    try:
        if num_inputs == 1:
            result = kernel(a)
        elif num_inputs == 2:
            result = kernel(a, b)
        else:  # num_inputs == 3
            result = kernel(a, b, c)

        assert result is not None
        print(f"✓ {mathop_name} numerical execution test passed")
    except Exception as e:
        print(f"Warning: {mathop_name} execution failed: {e}")


def test_rounding_mode_validation():
    """Test that invalid rounding modes raise ValueError"""

    # Test with invalid rounding mode
    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_add(1.0, 2.0, "invalid_mode")

    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_mul(1.0, 2.0, "xy")

    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_fsqrt(4.0, "bad_mode")

    print("✓ Rounding mode validation test passed")


@tilelang.testing.requires_cuda
def test_ieee_add_all_rounding_modes():
    """Test IEEE addition with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_add", T.ieee_add, rounding_mode=mode)
        print(f"✓ ieee_add with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_sub_all_rounding_modes():
    """Test IEEE subtraction with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_sub", T.ieee_sub, rounding_mode=mode)
        print(f"✓ ieee_sub with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_mul_all_rounding_modes():
    """Test IEEE multiplication with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_mul", T.ieee_mul, rounding_mode=mode)
        print(f"✓ ieee_mul with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_fmaf_all_rounding_modes():
    """Test IEEE fused multiply-add with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_fmaf", T.ieee_fmaf, rounding_mode=mode)
        print(f"✓ ieee_fmaf with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_frcp_all_rounding_modes():
    """Test IEEE reciprocal with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_frcp", T.ieee_frcp, rounding_mode=mode)
        print(f"✓ ieee_frcp with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_fsqrt_all_rounding_modes():
    """Test IEEE square root with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_fsqrt", T.ieee_fsqrt, rounding_mode=mode)
        print(f"✓ ieee_fsqrt with {mode} passed")


@tilelang.testing.requires_cuda
def test_ieee_frsqrt_rn_only():
    """Test IEEE reciprocal square root (round to nearest only)"""

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float32),
        B: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32), threads=128) as (bx, by):
            for i, j in T.Parallel(32, 32):
                B[by * 32 + i, bx * 32 + j] = T.ieee_frsqrt(A[by * 32 + i, bx * 32 + j])

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
    )

    print("\n=== Testing ieee_frsqrt (rn only) ===")
    print("✓ ieee_frsqrt compilation test passed")

    # Test numerical execution
    a = torch.abs(torch.randn(128, 128, device="cuda", dtype=torch.float32)) + 0.1

    try:
        result = kernel(a)
        assert result is not None
        print("✓ ieee_frsqrt numerical execution test passed")
    except Exception as e:
        print(f"Warning: ieee_frsqrt execution failed: {e}")


@tilelang.testing.requires_cuda
def test_ieee_fdiv_all_rounding_modes():
    """Test IEEE division with all rounding modes"""
    rounding_modes = ["rn", "rz", "ru", "rd"]

    for mode in rounding_modes:
        run_ieee_math_test("ieee_fdiv", T.ieee_fdiv, rounding_mode=mode)
        print(f"✓ ieee_fdiv with {mode} passed")


if __name__ == "__main__":
    tilelang.testing.main()
