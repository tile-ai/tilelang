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
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
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
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
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
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
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


# ===========================================================================
# rbits invariance -> chosen lane count
# ===========================================================================
#
# loop_vectorize.cc shrinks the cast vector size until the `rbits` operand
# is invariant across the chunk, so that one PTX cvt.rs.x{N} instruction can
# share a single rbits across N lanes. These tests pin that behaviour for
# the five rbits shapes that come up in practice.


@tilelang.jit
def _kernel_rbits_scalar(M: int, num_threads: int, target_dtype: str):
    """rbits = scalar fragment (loop-invariant) -> max lanes."""

    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            rbits_scalar = T.alloc_fragment((1,), "int32")
            T.copy(A, A_local)
            rbits_scalar[0] = T.cast(T.int32(0x12345678), "int32")
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=rbits_scalar[0])
            T.copy(B_local, B)

    return main


@tilelang.jit
def _kernel_rbits_idiv4(M: int, num_threads: int, target_dtype: str):
    """rbits = rbits_buf[i // 4] -> shared across 4 lanes, expect x4."""

    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            rbits_buf = T.alloc_fragment((M,), "int32")
            T.copy(A, A_local)
            for j in T.Parallel(M):
                rbits_buf[j] = T.cast(0x11111111, "int32")
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=rbits_buf[i // 4])
            T.copy(B_local, B)

    return main


@tilelang.jit
def _kernel_rbits_idiv2(M: int, num_threads: int, target_dtype: str):
    """rbits = rbits_buf[i // 2] -> shared across 2 lanes, expect x2."""

    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            rbits_buf = T.alloc_fragment((M,), "int32")
            T.copy(A, A_local)
            for j in T.Parallel(M):
                rbits_buf[j] = T.cast(0x22222222, "int32")
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=rbits_buf[i // 2])
            T.copy(B_local, B)

    return main


@tilelang.jit
def _kernel_rbits_per_element(M: int, num_threads: int, target_dtype: str):
    """rbits = rbits_buf[i] (per-element) -> no sharing, expect x1."""

    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            rbits_buf = T.alloc_fragment((M,), "int32")
            T.copy(A, A_local)
            for j in T.Parallel(M):
                rbits_buf[j] = T.cast(0x33333333, "int32")
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=rbits_buf[i])
            T.copy(B_local, B)

    return main


@tilelang.jit
def _kernel_rbits_rng_rand(M: int, num_threads: int, target_dtype: str):
    """rbits = T.rng_rand() -> curand call carries side effects, expect x1.

    Each lane needs its own random sample (PTX cvt.rs semantics shares one
    rbits across the chunk, which is the *opposite* of what callers want
    when they ask for fresh randomness per element). The vectorizer sees
    the side effect and falls back to lanes=1.
    """

    @T.prim_func
    def main(
        A: T.Tensor[(M,), "float32"],  # noqa: F821
        B: T.Tensor[(M,), target_dtype],  # noqa: F821
    ):
        with T.Kernel(1, threads=num_threads):
            A_local = T.alloc_fragment((M,), "float32")
            B_local = T.alloc_fragment((M,), target_dtype)
            T.copy(A, A_local)
            T.rng_init(42)
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], target_dtype, round="rs", rbits=T.rng_rand())
            T.copy(B_local, B)

    return main


_RBITS_BUILDERS = {
    "scalar": _kernel_rbits_scalar,
    "idiv4": _kernel_rbits_idiv4,
    "idiv2": _kernel_rbits_idiv2,
    "per_element": _kernel_rbits_per_element,
    "rng_rand": _kernel_rbits_rng_rand,
}

# Lane count is determined by rbits invariance, not by dtype, so fp8 and
# fp4 share the same expected lane table.
_RBITS_EXPECTED_LANES = {
    "scalar": 4,
    "idiv4": 4,
    "idiv2": 2,
    "per_element": 1,
    "rng_rand": 1,
}

_DTYPE_HELPER_TAG = {
    "float8_e4m3fn": "e4m3",
    "float4_e2m1fn": "e2m1",
}


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float4_e2m1fn"])
@pytest.mark.parametrize("rbits_mode", list(_RBITS_BUILDERS))
def test_cast_rs_rbits_invariance(rbits_mode, dtype):
    """rbits shape determines the chunk size of cvt.rs.x{N}."""
    M, threads = 128, 32
    builder = _RBITS_BUILDERS[rbits_mode]
    kernel = builder(M, threads, dtype)
    code = kernel.get_kernel_source()
    n = _RBITS_EXPECTED_LANES[rbits_mode]
    tag = _DTYPE_HELPER_TAG[dtype]
    expected = f"__tl_cvt_f32x{n}_to_{tag}x{n}_rs_sat"
    assert expected in code, f"Expected '{expected}' for rbits_mode={rbits_mode}, dtype={dtype}.\nGenerated code:\n{code}"


if __name__ == "__main__":
    tilelang.testing.main()
