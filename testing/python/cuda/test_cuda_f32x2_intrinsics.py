"""Tests for packed x2 intrinsics (add2, sub2, mul2, fma2, max2, min2, abs2).

Each operation is tested for all three supported dtype families:
  - float32   (float32x2)
  - bfloat16  (bfloat16x2)
  - float16   (float16x2)

Three kinds of tests:
  1. Codegen tests  -- verify that the CUDA source contains ``tl::<op>``
     and that bfloat16x2/float16x2 emit proper native-type casts
     (__nv_bfloat162 / __half2) instead of the ambiguous uint1 overload.
  2. Correctness tests -- compile, run, and compare against PyTorch reference.
  3. Auto-vectorization tests -- verify SM100 auto-vectorization behaviour.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import pytest
import torch

SM100_TARGET = "cuda -arch=sm_100"
SM80_TARGET = "cuda -arch=sm_80"

M = 128  # number of threads / element-pairs

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"float32": (T.float32, torch.float32), "bfloat16": (T.bfloat16, torch.bfloat16), "float16": (T.float16, torch.float16)}

# ---------------------------------------------------------------------------
# Generic kernel builders using T.Ramp for packed x2 access
# ---------------------------------------------------------------------------


def _make_binary_kernel(op_func, dtype_tl):
    """Build a kernel: C[idx] = op(A[idx], B[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx], B[idx])

    return main


def _make_ternary_kernel(op_func, dtype_tl):
    """Build a kernel: D[idx] = op(A[idx], B[idx], C[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
        D: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            D[idx] = op_func(A[idx], B[idx], C[idx])

    return main


def _make_unary_kernel(op_func, dtype_tl):
    """Build a kernel: C[idx] = op(A[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx])

    return main


# ---------------------------------------------------------------------------
# Helper: lower to CUDA source
# ---------------------------------------------------------------------------


def _lower_to_cuda_source(func, target: str = SM80_TARGET) -> str:
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


# ---------------------------------------------------------------------------
# Auto-vectorization kernels via T.Parallel
# ---------------------------------------------------------------------------

# Map from Python operator string to (lambda, tl_func_name)
_AUTO_VEC_OPS = {"add": (lambda a, b: a + b, "add2"), "sub": (lambda a, b: a - b, "sub2"), "mul": (lambda a, b: a * b, "mul2")}


def _make_auto_vec_binary_kernel(py_op, dtype_tl, width: int = 4):
    """Build a kernel that uses T.Parallel to let the vectoriser emit tl::<op>2."""

    @T.prim_func
    def main(
        A: T.Tensor((M, width), dtype=dtype_tl),
        B: T.Tensor((M, width), dtype=dtype_tl),
        C: T.Tensor((M, width), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, width):
                C[i, v] = py_op(A[i, v], B[i, v])

    return main


def _make_auto_vec_fma_kernel(dtype_tl, width: int = 4):
    """Build a kernel that lets CUDA codegen fuse mul + add into tl::fma2."""

    @T.prim_func
    def main(
        A: T.Tensor((M, width), dtype=dtype_tl),
        B: T.Tensor((M, width), dtype=dtype_tl),
        C: T.Tensor((M, width), dtype=dtype_tl),
        D: T.Tensor((M, width), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, width):
                D[i, v] = A[i, v] * B[i, v] + C[i, v]

    return main


# ===================================================================
# Parametrised op / dtype lists
# ===================================================================

# Binary ops: (name, func)
_BINARY_OPS = [
    ("add2", T.add2),
    ("sub2", T.sub2),
    ("mul2", T.mul2),
    ("max2", T.max2),
    ("min2", T.min2),
]

# All 3 dtype families
_DTYPES = ["float32", "bfloat16", "float16"]

# Native cast types expected in codegen for 16-bit packed types
_NATIVE_CAST_TYPE = {"bfloat16": "__nv_bfloat162", "float16": "__half2"}

# Torch reference functions
_TORCH_REFS = {
    "add2": lambda a, b: a + b,
    "sub2": lambda a, b: a - b,
    "mul2": lambda a, b: a * b,
    "max2": lambda a, b: torch.maximum(a, b),
    "min2": lambda a, b: torch.minimum(a, b),
    "fma2": lambda a, b, c: a * b + c,
    "abs2": lambda a: torch.abs(a),
}


# ===================================================================
# Codegen tests
# ===================================================================


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[n for n, _ in _BINARY_OPS])
def test_codegen_binary(op_name, op_func, dtype_name):
    """Binary ops emit tl::<op> with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_binary_kernel(op_func, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{op_name}" in src, f"Expected tl::{op_name} in generated CUDA source"
    # For 16-bit types, verify that the codegen emits casts to the correct
    # native type instead of the ambiguous uint1 overload.
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_codegen_fma2(dtype_name):
    """fma2 emits tl::fma2 with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_ternary_kernel(T.fma2, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::fma2" in src
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_codegen_abs2(dtype_name):
    """abs2 emits tl::abs2 with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_unary_kernel(T.abs2, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::abs2" in src
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


# ---------------------------------------------------------------------------
# Auto-vectorization codegen tests (T.Parallel -> tl::*2)
# ---------------------------------------------------------------------------

_AUTO_VEC_OP_NAMES = list(_AUTO_VEC_OPS.keys())  # ["add", "sub", "mul"]


# float32: auto-vectorization should emit tl::<op>2 on SM100+
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert f"tl::{tl_func}" in src, f"Expected tl::{tl_func} in SM100 auto-vectorised CUDA source for float32 {op_key}"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32_width8(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32, width=8)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert "\x00" not in src, "Generated CUDA source should not contain embedded NUL bytes"
    for field in "xyzw":
        assert f".{field})) = tl::{tl_func}(" in src, (
            f"Expected {field}-field packed tl::{tl_func} emission in width-8 float32 auto-vectorised source"
        )


# float32: auto-vectorization should NOT emit tl::<op>2 before SM100
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32_no_sm80(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{tl_func}" not in src, f"tl::{tl_func} should NOT appear in SM80 auto-vectorised CUDA source for float32 {op_key}"


@tilelang.testing.requires_cuda
def test_codegen_auto_vec_fma_f32():
    func = _make_auto_vec_fma_kernel(T.float32)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert "tl::fma2" in src, "Expected tl::fma2 in SM100 auto-vectorised CUDA source for float32 mul+add"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_codegen_auto_vec_fma_half_types(dtype_name):
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_auto_vec_fma_kernel(dtype_tl, width=8)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::fma2" in src, f"Expected tl::fma2 in CUDA source for {dtype_name} mul+add"
    assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


# bfloat16 / float16: auto-vectorization should emit tl::<op>2 on any target
# (the C++ helpers have compile-time arch fallbacks).
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_half_types(op_key, dtype_name):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_auto_vec_binary_kernel(py_op, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{tl_func}" in src, f"Expected tl::{tl_func} in auto-vectorised CUDA source for {dtype_name} {op_key}"
    # Verify correct native-type cast
    assert _NATIVE_CAST_TYPE[dtype_name] in src, (
        f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in auto-vectorised CUDA source for {dtype_name} {op_key}"
    )


# ===================================================================
# Numerical correctness tests
# ===================================================================


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[n for n, _ in _BINARY_OPS])
def test_correctness_binary(op_name, op_func, dtype_name):
    """Binary ops produce correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_binary_kernel(op_func, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    b = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_name](a, b)
    torch.testing.assert_close(c, ref)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_correctness_fma2(dtype_name):
    """fma2 produces correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_ternary_kernel(T.fma2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[3], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    b = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    d = kernel(a, b, c)
    ref = _TORCH_REFS["fma2"](a, b, c)
    # Hardware FMA fuses multiply-add into a single rounding step, so it can
    # differ from the separate mul+add reference by up to 1 ULP.  Use relaxed
    # tolerances for 16-bit types.
    if dtype_name == "float32":
        torch.testing.assert_close(d, ref)
    else:
        torch.testing.assert_close(d, ref, atol=1e-2, rtol=1e-1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_correctness_abs2(dtype_name):
    """abs2 produces correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_unary_kernel(T.abs2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = kernel(a)
    ref = _TORCH_REFS["abs2"](a)
    torch.testing.assert_close(c, ref)


if __name__ == "__main__":
    tilelang.testing.main()
