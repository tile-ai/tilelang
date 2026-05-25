"""Tests for T.cast with round, sat, and rbits parameters."""

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.language.tir.ir import _VALID_CAST_ROUNDING_MODES

# T.cast is provided by tilelang.language.tir.ir (and re-exported by
# `from .tir.ir import *` in tilelang.language.__init__, overriding the
# upstream T.cast brought in by `from tvm.script.parser.tir import *`).
cast = T.cast


# ===========================================================================
# Validation tests (do not require GPU)
# ===========================================================================


def test_cast_invalid_round():
    """Test that invalid rounding modes raise ValueError."""
    with pytest.raises(ValueError, match="Invalid round"):
        cast(1.0, "float8_e4m3fn", round="invalid")


def test_cast_invalid_sat():
    """Test that non-bool sat raises ValueError."""
    with pytest.raises(ValueError, match="Invalid sat"):
        cast(1.0, "float8_e4m3fn", sat="invalid")


def test_cast_rs_requires_rbits():
    """Test that round='rs' requires rbits."""
    with pytest.raises(ValueError, match="rbits is required"):
        cast(1.0, "float8_e4m3fn", round="rs")


def test_cast_rbits_only_with_rs():
    """Test that rbits is only valid with round='rs'."""
    with pytest.raises(ValueError, match="rbits is only valid"):
        cast(1.0, "float8_e4m3fn", round="rn", rbits=T.int32(42))


def test_cast_valid_rounding_modes():
    """Test that all valid rounding modes are accepted (without rbits for non-rs)."""
    for mode in _VALID_CAST_ROUNDING_MODES:
        if mode in ("rs", ""):
            continue
        # Should not raise
        cast(T.float32(1.0), "float8_e4m3fn", round=mode)


def test_cast_sat_bool():
    """Test that sat accepts True and False."""
    # Should not raise
    cast(T.float32(1.0), "float8_e4m3fn", sat=True)
    cast(T.float32(1.0), "float8_e4m3fn", sat=False)


# ===========================================================================
# Codegen tests (require GPU with compute capability >= 8.9)
# ===========================================================================


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
def test_cast_default_unchanged():
    """Test that T.cast without rounding params still uses __nv_cvt_float2_to_fp8x2."""
    M = 256

    @tilelang.jit
    def default_cast_kernel(M_val: int):
        @T.prim_func
        def main(
            A: T.Tensor[(M_val,), "float32"],  # noqa: F821
            B: T.Tensor[(M_val,), "float8_e4m3fn"],  # noqa: F821
        ):
            with T.Kernel(1, threads=128):
                A_local = T.alloc_fragment((M_val,), "float32")
                B_local = T.alloc_fragment((M_val,), "float8_e4m3fn")
                T.copy(A, A_local)
                for i in T.Parallel(M_val):
                    B_local[i] = T.cast(A_local[i], "float8_e4m3fn")
                T.copy(B_local, B)

        return main

    kernel = default_cast_kernel(M)
    code = kernel.get_kernel_source()
    assert "__nv_cvt_float2_to_fp8x2" in code, (
        f"Default cast should use __nv_cvt_float2_to_fp8x2 for backward compatibility.\nGenerated code:\n{code}"
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
def test_cast_rs_codegen():
    """Test that round='rs' with rbits emits __tl_cvt_f32x4_to_e4m3x4_rs_sat."""
    M = 256

    @tilelang.jit
    def cast_rs_kernel(M_val: int):
        @T.prim_func
        def main(
            A: T.Tensor[(M_val,), "float32"],  # noqa: F821
            B: T.Tensor[(M_val,), "float8_e4m3fn"],  # noqa: F821
        ):
            with T.Kernel(1, threads=128):
                A_local = T.alloc_fragment((M_val,), "float32")
                B_local = T.alloc_fragment((M_val,), "float8_e4m3fn")
                rbits = T.alloc_fragment((1,), "int32")
                T.copy(A, A_local)
                rbits[0] = T.cast(T.int32(0x12345678), "int32")
                for i in T.Parallel(M_val):
                    B_local[i] = T.cast(A_local[i], "float8_e4m3fn", round="rs", rbits=rbits[0])
                T.copy(B_local, B)

        return main

    kernel = cast_rs_kernel(M)
    code = kernel.get_kernel_source()
    # lanes=2 after vectorization -> uses x2 variant with zero-padding
    assert "__tl_cvt_f32x2_to_e4m3x2_rs_sat" in code or "__tl_cvt_f32x4_to_e4m3x4_rs_sat" in code, (
        f"Expected stochastic rounding helper in generated code.\nGenerated code:\n{code}"
    )


# ===========================================================================
# Multi-lanes rs codegen tests (FP8 and FP4)
# ===========================================================================


@tilelang.jit
def _make_rs_kernel(M: int, num_threads: int, target_dtype: str):
    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            rbits = T.alloc_fragment((1,), "int32")
            T.copy(A, A_local)
            rbits[0] = T.cast(T.int32(0x12345678), "int32")
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=rbits[0])
            T.copy(B_local, B)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize(
    "M,threads,expected",
    [
        (128, 128, "__tl_cvt_f32x1_to_e4m3x1_rs_sat"),
        (256, 128, "__tl_cvt_f32x2_to_e4m3x2_rs_sat"),
        (512, 128, "__tl_cvt_f32x4_to_e4m3x4_rs_sat"),
        (1024, 128, "__tl_cvt_f32x4_to_e4m3x4_rs_sat"),
    ],
)
def test_cast_rs_fp8_lanes(M, threads, expected):
    """Test FP8 rs codegen across lanes=1,2,4,8."""
    kernel = _make_rs_kernel(M, threads, "float8_e4m3fn")
    code = kernel.get_kernel_source()
    assert expected in code, f"Expected '{expected}' in generated code for M={M}, threads={threads}.\nGenerated code:\n{code}"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize(
    "M,threads,expected",
    [
        (128, 128, "__tl_cvt_f32x1_to_e2m1x1_rs_sat"),
        (256, 128, "__tl_cvt_f32x2_to_e2m1x2_rs_sat"),
        (512, 128, "__tl_cvt_f32x4_to_e2m1x4_rs_sat"),
        (1024, 128, "__tl_cvt_f32x4_to_e2m1x4_rs_sat"),
    ],
)
def test_cast_rs_fp4_lanes(M, threads, expected):
    """Test FP4 rs codegen across lanes=1,2,4,8."""
    kernel = _make_rs_kernel(M, threads, "float4_e2m1fn")
    code = kernel.get_kernel_source()
    assert expected in code, f"Expected '{expected}' in generated code for M={M}, threads={threads}.\nGenerated code:\n{code}"


if __name__ == "__main__":
    tilelang.testing.main()
