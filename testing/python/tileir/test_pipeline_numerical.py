"""Numerical correctness tests for the TileIR pipeline.

Verifies that the default ``build_tileir_module`` pipeline produces GPU kernels
whose outputs match PyTorch references within tight tolerances.

Kernels under test:
  - fill        : T.fill(buf, value)    → compare to torch.full_like
  - copy        : T.copy(A → shared → B) → compare to identity copy
  - tiny_gemm   : T.gemm(sa, sb, acc), f16 in / f32 out
  - gemv        : T.gemm(sa, sb, acc, transpose_B=True), f16 in / f16 out
  - scalar      : elementwise multiply with a scalar entry parameter

The tests use ``tilelang.compile(prim_func, execution_backend="tileir")`` so the
production default path is exercised directly.
"""

from __future__ import annotations

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

# ---------------------------------------------------------------------------
# Guard: cuda_tile MLIR bindings required to build any kernel
# ---------------------------------------------------------------------------

try:
    from cuda_tile._mlir import ir as _ir  # noqa: F401

    _HAS_CUDA_TILE = True
except ImportError:
    _HAS_CUDA_TILE = False

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILEIR_PASS_CONFIGS = {}


def _skip_if_tileir_toolchain_unavailable():
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def _setup_gpu():
    """Return (torch, major, minor, target_str) or skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    major, minor = torch.cuda.get_device_capability()
    target_str = f"tileir -arch=sm_{major}{minor}"
    return torch, major, minor, target_str


def _compile_and_run(prim_func, target_str, out_idx, pass_configs, *inputs):
    """Compile prim_func through the TileIR pipeline and execute the inputs.

    Returns the output tensor(s) indicated by out_idx.
    """
    kernel = tilelang.compile(
        prim_func,
        out_idx=out_idx,
        execution_backend="tileir",
        target=target_str,
        pass_configs=pass_configs,
    )
    return kernel(*inputs)


# ---------------------------------------------------------------------------
# Test 1: fill
# ---------------------------------------------------------------------------

_FILL_M, _FILL_N, _FILL_VAL = 32, 64, 3.14


@T.prim_func
def _fill_prim_func(A: T.Tensor((_FILL_M, _FILL_N), "float32")):
    with T.Kernel(1, threads=128):
        T.fill(A, _FILL_VAL)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_fill_numerical(monkeypatch):
    """fill kernel: output must equal torch.full_like(A, val)."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A_in = torch.zeros(_FILL_M, _FILL_N, dtype=torch.float32, device="cuda")
    out = _compile_and_run(_fill_prim_func, target_str, [-1], TILEIR_PASS_CONFIGS, A_in)

    ref = torch.full((_FILL_M, _FILL_N), _FILL_VAL, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: copy
# ---------------------------------------------------------------------------

_COPY_M, _COPY_N = 32, 64


@T.prim_func
def _copy_prim_func(A: T.Tensor((_COPY_M, _COPY_N), "float16"), B: T.Tensor((_COPY_M, _COPY_N), "float16")):
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared([_COPY_M, _COPY_N], T.float16)
        T.copy(A, shared)
        T.copy(shared, B)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_copy_numerical(monkeypatch):
    """copy kernel: output must equal the input (identity copy)."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_COPY_M, _COPY_N, dtype=torch.float16, device="cuda")
    out = _compile_and_run(_copy_prim_func, target_str, [-1], TILEIR_PASS_CONFIGS, A)

    torch.testing.assert_close(out, A, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 3: tiny_gemm
# ---------------------------------------------------------------------------

_GEMM_M, _GEMM_K, _GEMM_N = 64, 64, 64


@T.prim_func
def _tiny_gemm_prim_func(
    A: T.Tensor((_GEMM_M, _GEMM_K), "float16"),
    B: T.Tensor((_GEMM_K, _GEMM_N), "float16"),
    C: T.Tensor((_GEMM_M, _GEMM_N), "float32"),
):
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([_GEMM_M, _GEMM_K], T.float16)
        sb = T.alloc_shared([_GEMM_K, _GEMM_N], T.float16)
        acc = T.alloc_fragment([_GEMM_M, _GEMM_N], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.gemm(sa, sb, acc)
        T.copy(acc, C)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_tiny_gemm_numerical(monkeypatch):
    """tiny_gemm: C = A @ B, f16 inputs, f32 output."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_GEMM_M, _GEMM_K, dtype=torch.float16, device="cuda")
    B = torch.randn(_GEMM_K, _GEMM_N, dtype=torch.float16, device="cuda")
    out = _compile_and_run(_tiny_gemm_prim_func, target_str, [-1], TILEIR_PASS_CONFIGS, A, B)

    ref = A.float() @ B.float()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Test 4: gemv (transpose_B=True)
# ---------------------------------------------------------------------------

_GEMV_M, _GEMV_K, _GEMV_N = 64, 64, 64


@T.prim_func
def _gemv_prim_func(
    A: T.Tensor((_GEMV_M, _GEMV_K), "float16"),
    B: T.Tensor((_GEMV_N, _GEMV_K), "float16"),
    C: T.Tensor((_GEMV_M, _GEMV_N), "float16"),
):
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([_GEMV_M, _GEMV_K], T.float16)
        sb = T.alloc_shared([_GEMV_N, _GEMV_K], T.float16)
        acc = T.alloc_fragment([_GEMV_M, _GEMV_N], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.gemm(sa, sb, acc, transpose_B=True)
        T.copy(acc, C)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_gemv_numerical(monkeypatch):
    """gemv: C = A @ B^T, f16 in/out (acc in f32, cast to f16)."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_GEMV_M, _GEMV_K, dtype=torch.float16, device="cuda")
    B = torch.randn(_GEMV_N, _GEMV_K, dtype=torch.float16, device="cuda")
    out = _compile_and_run(_gemv_prim_func, target_str, [-1], TILEIR_PASS_CONFIGS, A, B)

    ref = (A.float() @ B.float().t()).half()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Test 5: scalar param — scale multiply
# ---------------------------------------------------------------------------

_SCALAR_M, _SCALAR_N = 32, 64


@T.prim_func
def _scalar_param_prim_func(
    A: T.Tensor((_SCALAR_M, _SCALAR_N), "float32"),
    B: T.Tensor((_SCALAR_M, _SCALAR_N), "float32"),
    scale: T.float32,
):
    """B[i, j] = A[i, j] * scale — single scalar entry param."""
    with T.Kernel(1, threads=128):
        for i, j in T.Parallel(_SCALAR_M, _SCALAR_N):
            B[i, j] = A[i, j] * scale


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_scalar_param_numerical(monkeypatch):
    """scalar param: B = A * scale, f32 I/O; verifies scalar entry-arg binding."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_SCALAR_M, _SCALAR_N, dtype=torch.float32, device="cuda")
    scale_val = 3.14
    scale_tensor = torch.tensor(scale_val, dtype=torch.float32, device="cuda")
    out = _compile_and_run(
        _scalar_param_prim_func,
        target_str,
        [-2],
        TILEIR_PASS_CONFIGS,
        A,
        scale_tensor,
    )

    ref = A * scale_val
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
