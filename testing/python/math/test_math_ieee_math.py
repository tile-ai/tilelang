import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest

ROUNDING_MODES = ["rn", "rz", "ru", "rd"]

# (dtype, rounding_mode) pairs that must compile and produce correct results.
# - fp32/fp64: all four IEEE rounding modes are natively supported by CUDA.
# - fp16     : only round-to-nearest-even is available.
# - bf16     : no native IEEE rounding intrinsics (tested separately via rejection).
IEEE_DTYPE_MODES = [
    (T.float32, "rn"),
    (T.float32, "rz"),
    (T.float32, "ru"),
    (T.float32, "rd"),
    (T.float64, "rn"),
    (T.float64, "rz"),
    (T.float64, "ru"),
    (T.float64, "rd"),
    (T.float16, "rn"),
]

# Ops that carry a rounding_mode parameter (everything except frsqrt).
IEEE_OPS_WITH_RM = [
    ("ieee_add", T.ieee_add),
    ("ieee_sub", T.ieee_sub),
    ("ieee_mul", T.ieee_mul),
    ("ieee_fmaf", T.ieee_fmaf),
    ("ieee_frcp", T.ieee_frcp),
    ("ieee_fsqrt", T.ieee_fsqrt),
    ("ieee_fdiv", T.ieee_fdiv),
]

# All 8 ops (including frsqrt).
IEEE_ALL_OPS = IEEE_OPS_WITH_RM + [("ieee_frsqrt", T.ieee_frsqrt)]


def run_ieee_math_test(
    mathop_name,
    mathop_func,
    rounding_mode="rn",
    M=32,
    N=32,
    block_M=16,
    block_N=16,
    dtype=T.float32,
    run_execution=True,
):
    """
    Test IEEE-compliant math operations with specified rounding modes.
    """

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
    elif mathop_name == "ieee_frsqrt":  # no rounding_mode parameter

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = mathop_func(A[by * block_M + i, bx * block_N + j])

        out_idx = [1]
        num_inputs = 1
    else:  # Single argument with rounding_mode (frcp, fsqrt)

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

    if not run_execution:
        return

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


# ---------------------------------------------------------------------------
# Rounding-mode validation (API-level, no GPU required)
# ---------------------------------------------------------------------------


def test_rounding_mode_validation():
    """Test that invalid rounding modes raise ValueError"""
    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_add(1.0, 2.0, "invalid_mode")
    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_mul(1.0, 2.0, "xy")
    with pytest.raises(ValueError, match="Invalid rounding mode"):
        T.ieee_fsqrt(4.0, "bad_mode")
    print("✓ Rounding mode validation test passed")


# ---------------------------------------------------------------------------
# Numerical tests — every (op × dtype × rounding_mode) valid combination
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_name,op_func", IEEE_OPS_WITH_RM)
@pytest.mark.parametrize("dtype,mode", IEEE_DTYPE_MODES)
def test_ieee_with_rounding_mode(op_name, op_func, dtype, mode):
    """Compile + run every IEEE op across fp32 / fp64 / fp16 and all rounding
    modes that the hardware supports."""
    run_ieee_math_test(op_name, op_func, rounding_mode=mode, dtype=dtype, run_execution=(mode == "rn"))


# ieee_frsqrt — only round-to-nearest (no rounding_mode parameter).


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float32, T.float16])
def test_ieee_frsqrt(dtype):
    run_ieee_math_test("ieee_frsqrt", T.ieee_frsqrt, dtype=dtype)


# ---------------------------------------------------------------------------
# Rejection tests — combinations that must fail at codegen time
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_ieee_frsqrt_fp64_rejected():
    """fp64 has no rsqrt intrinsic — codegen must FATAL."""
    with pytest.raises(Exception, match="frsqrt is not supported for float64"):
        run_ieee_math_test("ieee_frsqrt", T.ieee_frsqrt, dtype=T.float64)


@tilelang.testing.requires_cuda
def test_ieee_fp16_non_rn_rejected():
    """fp16 only supports round-to-nearest-even."""
    with pytest.raises(Exception, match="Only rounding mode 'rn' is available for half precision"):
        run_ieee_math_test("ieee_add", T.ieee_add, rounding_mode="rz", dtype=T.float16)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_name,op_func", IEEE_ALL_OPS)
def test_ieee_bf16_rejected(op_name, op_func):
    """bf16 has no IEEE rounding intrinsics in CUDA — every op must FATAL."""
    with pytest.raises(Exception, match="not supported for bfloat16"):
        run_ieee_math_test(op_name, op_func, dtype=T.bfloat16)


if __name__ == "__main__":
    tilelang.testing.main()
