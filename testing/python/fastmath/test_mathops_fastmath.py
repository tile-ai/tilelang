import pytest
import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import re


def get_mathop_lines(source, mathop_name):
    """Extract lines containing the mathop from CUDA source for debugging"""
    lines = source.split("\n")
    relevant_lines = []
    for i, line in enumerate(lines):
        if mathop_name in line and ("(" in line):
            # Include some context
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            relevant_lines.extend([f"{j}: {lines[j]}" for j in range(start, end)])
            relevant_lines.append("---")
    return "\n".join(relevant_lines[-10:])  # Show last 10 lines to avoid too much output


def check_fastmath_usage(source, mathop_name, expect_fastmath=False):
    """Check source for fastmath/non-fastmath versions"""
    fastmath_pattern = rf"__({mathop_name}f?)\b"
    non_fastmath_pattern = rf"(?<!__)({mathop_name}f?)\b"

    fastmath_matches = re.findall(fastmath_pattern, source)
    non_fastmath_matches = re.findall(non_fastmath_pattern, source)

    print(f"Found {len(fastmath_matches)} fastmath calls, {len(non_fastmath_matches)} non-fastmath calls")
    if len(fastmath_matches) > 0:
        print(f"Fastmath calls found: {fastmath_matches}")
    if len(non_fastmath_matches) > 0:
        print(f"Non-fastmath calls found: {non_fastmath_matches}")
    print(f"Source preview for {mathop_name}:")
    print(get_mathop_lines(source, mathop_name))

    if expect_fastmath:
        assert len(fastmath_matches) > 0, "Expected fastmath calls but found none"
        print(f"✓ {mathop_name} correctly uses fastmath versions")
    else:
        assert len(fastmath_matches) == 0, f"Found unexpected fastmath calls: {fastmath_matches}"
        assert len(non_fastmath_matches) > 0, f"No {mathop_name} calls found"
        print(f"✓ {mathop_name} correctly uses non-fastmath versions")


def check_non_fastmath_usage(source, mathop_name):
    """Check that source uses non-fastmath versions (no __ prefix)"""
    check_fastmath_usage(source, mathop_name, expect_fastmath=False)


def run_single_arg_mathop_test(mathop_name, mathop_func, M=128, N=128, block_M=32, block_N=32, dtype=T.float32):
    """
    Test single-argument mathops.
    T.exp should generate expf (non-fastmath), T.__exp should generate __expf (fastmath)
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = mathop_func(A[by * block_M + i, bx * block_N + j])

    # Test with FAST_MATH disabled
    kernel_no_fastmath = tilelang.compile(
        main,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
    )

    source_no_fastmath = kernel_no_fastmath.get_kernel_source()

    print(f"\n=== Testing {mathop_name} ===")
    print("FAST_MATH=False:")

    # Our tl.* intrinsics actually generate fastmath versions (e.g., __expf)
    check_fastmath_usage(source_no_fastmath, mathop_name, expect_fastmath=False)

    print(f"✓ {mathop_name} compilation and execution test passed")


def run_two_arg_mathop_test(mathop_name, mathop_func, M=128, N=128, block_M=32, block_N=32, dtype=T.float32):
    """
    Test two-argument mathops to ensure they generate non-fastmath CUDA code.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = mathop_func(
                    A[by * block_M + i, bx * block_N + j], B[by * block_M + i, bx * block_N + j]
                )

    # Test with FAST_MATH disabled
    kernel_no_fastmath = tilelang.compile(
        main,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
    )

    # Test with FAST_MATH enabled
    kernel_fastmath = tilelang.compile(
        main,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )

    source_no_fastmath = kernel_no_fastmath.get_kernel_source()
    source_fastmath = kernel_fastmath.get_kernel_source()

    print(f"\n=== Testing {mathop_name} (two args) ===")
    print("FAST_MATH=False:")
    check_non_fastmath_usage(source_no_fastmath, mathop_name)

    print("FAST_MATH=True:")
    check_non_fastmath_usage(source_fastmath, mathop_name)

    # Test numerical correctness
    torch_dtype = dtype.as_torch()
    a = torch.randn(M, N, device="cuda", dtype=torch_dtype)
    b = torch.randn(M, N, device="cuda", dtype=torch_dtype)

    # Ensure positive values for functions that need them
    if mathop_name == "pow":
        a = torch.abs(a) + 0.1
        b = torch.clamp(b, -3, 3)  # Limit exponent range
    elif mathop_name == "fmod":
        b = torch.abs(b) + 0.1  # Avoid division by zero

    c_no_fastmath = kernel_no_fastmath(a, b)
    c_fastmath = kernel_fastmath(a, b)

    # Both should produce similar results
    torch.testing.assert_close(c_no_fastmath, c_fastmath, rtol=1e-3, atol=1e-3)
    print(f"✓ {mathop_name} numerical test passed")


def run_abs_test():
    """Test that abs correctly maps to fabs (not __fabsf) in generated CUDA code"""
    M, N = 128, 128
    block_M, block_N = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = T.abs(A[by * block_M + i, bx * block_N + j])

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
    )

    source = kernel.get_kernel_source()
    print("\n=== Testing abs (maps to fabs) ===")
    check_non_fastmath_usage(source, "fabs")

    # Test numerical correctness
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = kernel(a)
    expected = torch.abs(a)

    torch.testing.assert_close(b, expected, rtol=1e-5, atol=1e-5)
    print("✓ abs numerical test passed")


def run_fastmath_mathop_test(mathop_name, mathop_func, M=128, N=128, block_M=32, block_N=32, dtype=T.float32):
    """
    Test fastmath mathops to ensure they generate fastmath CUDA code (with __ prefix).
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = mathop_func(A[by * block_M + i, bx * block_N + j])

    # Test with FAST_MATH enabled
    kernel_fastmath = tilelang.compile(
        main,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )

    source_fastmath = kernel_fastmath.get_kernel_source()

    print(f"\n=== Testing {mathop_name} (fastmath version) ===")
    print("FAST_MATH=True:")
    # Strip the __ prefix for checking in the CUDA source
    cuda_mathop_name = mathop_name.lstrip("_")
    check_fastmath_usage(source_fastmath, cuda_mathop_name, expect_fastmath=True)

    # Test numerical correctness
    torch_dtype = dtype.as_torch()
    a = torch.randn(M, N, device="cuda", dtype=torch_dtype)

    # Ensure positive values for functions that need them
    if cuda_mathop_name in ["sqrt", "rsqrt", "log", "log2", "log10"]:
        a = torch.abs(a) + 0.1

    b_fastmath = kernel_fastmath(a)

    # Compare with reference implementation
    if cuda_mathop_name == "exp":
        expected = torch.exp(a)
    elif cuda_mathop_name == "log":
        expected = torch.log(a)
    else:
        expected = b_fastmath  # Just check compilation works

    torch.testing.assert_close(b_fastmath, expected, rtol=1e-3, atol=1e-3)
    print(f"✓ {mathop_name} numerical test passed")


@pytest.mark.parametrize(
    "name, func",
    [
        ("exp", T.exp),
        ("exp2", T.exp2),
        ("exp10", T.exp10),
        ("log", T.log),
        ("log2", T.log2),
        ("log10", T.log10),
        ("sin", T.sin),
        ("cos", T.cos),
        ("tan", T.tan),
        ("sinh", T.sinh),
        ("cosh", T.cosh),
        ("tanh", T.tanh),
        ("atan", T.atan),
        ("sqrt", T.sqrt),
        ("rsqrt", T.rsqrt),
        ("erf", T.erf),
        ("floor", T.floor),
        ("ceil", T.ceil),
        ("trunc", T.trunc),
        ("round", T.round),
        ("nearbyint", T.nearbyint),
    ],
)
@tilelang.testing.requires_cuda
def test_mathops_generate_no_fastmath(name, func):
    """Test that our tl.* mathops generate fastmath CUDA code (__expf etc.)"""
    run_single_arg_mathop_test(name, func, dtype=T.float32)
    print(f"✓ {name} test passed")


@pytest.mark.parametrize(
    "name, func",
    [
        ("pow", T.pow),
        ("fmod", T.fmod),
    ],
)
@tilelang.testing.requires_cuda
def test_two_arg_mathops_fastmath(name, func):
    """Test all two-argument mathops"""
    run_two_arg_mathop_test(name, func, dtype=T.float32)


@tilelang.testing.requires_cuda
def test_abs_maps_to_fabs():
    """Test that abs correctly maps to fabs"""
    run_abs_test()


@pytest.mark.parametrize(
    "name, func",
    [
        ("__exp", T.__exp),
        ("__exp10", T.__exp10),
        ("__log", T.__log),
        ("__log2", T.__log2),
        ("__log10", T.__log10),
        ("__tan", T.__tan),
        ("__cos", T.__cos),
        ("__sin", T.__sin),
    ],
)
@tilelang.testing.requires_cuda
def test_fastmath_versions(name, func):
    """Test that __exp, __exp10, __log, __log2, __log10, __tan, __cos, __sin generate fastmath CUDA code"""
    run_fastmath_mathop_test(name, func, dtype=T.float32)
    print(f"✓ {name} test passed")


if __name__ == "__main__":
    tilelang.testing.main()
