"""Numerical tests for T.wgmma_gemm (tl.tileop.wgmma_gemm) on the TileIR backend.

`T.wgmma_gemm` (tilelang/language/gemm_op.py) shares `_gemm_impl`'s exact TIR
call layout with `T.gemm`; it only pins the Hopper WGMMA instruction and skips
the implicit warpgroup wait (wg_wait=-1, paired with an explicit
`T.wait_wgmma(id)`).  In the TileIR backend instruction selection belongs to
the downstream cuda_tile optimizer and GEMM completion is ordered by TKO
tokens, so:

  - tl.tileop.wgmma_gemm lowers as an alias of the plain Gemm lowering
    (tilelang/tileir/lowering/sem_to_ir/tile_ops.py), and
  - tl.wait_wgmma lowers to the no-op Barrier hint.

These tests mirror the usage in examples/deepseek_v32/sparse_mla_fwd_pipelined.py
(several wgmma_gemm accumulations into one fragment followed by
T.wait_wgmma(0)) and compare against torch references.
"""

from __future__ import annotations

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T

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

TILEIR_PASS_CONFIGS = {}


def _skip_if_tileir_toolchain_unavailable():
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def _setup_gpu():
    """Return (torch, target_str) or skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    major, minor = torch.cuda.get_device_capability()
    target_str = f"tileir -arch=sm_{major}{minor}"
    return torch, target_str


def _compile_and_run(prim_func, target_str, out_idx, *inputs):
    kernel = tilelang.compile(
        prim_func,
        out_idx=out_idx,
        execution_backend="tileir",
        target=target_str,
        pass_configs=TILEIR_PASS_CONFIGS,
    )
    return kernel(*inputs)


# ---------------------------------------------------------------------------
# Test 1: single wgmma_gemm, transpose_B=True, explicit wait_wgmma
# ---------------------------------------------------------------------------

_M, _K, _N = 64, 64, 64


@T.prim_func
def _wgmma_gemm_prim_func(
    A: T.Tensor((_M, _K), "float16"),
    B: T.Tensor((_N, _K), "float16"),
    C: T.Tensor((_M, _N), "float32"),
):
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([_M, _K], T.float16)
        sb = T.alloc_shared([_N, _K], T.float16)
        acc = T.alloc_fragment([_M, _N], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.wgmma_gemm(sa, sb, acc, transpose_B=True)
        T.wait_wgmma(0)
        T.copy(acc, C)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_wgmma_gemm_numerical(monkeypatch):
    """wgmma_gemm: C = A @ B^T, f16 inputs, f32 output."""
    _skip_if_tileir_toolchain_unavailable()
    torch, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_M, _K, dtype=torch.float16, device="cuda")
    B = torch.randn(_N, _K, dtype=torch.float16, device="cuda")
    out = _compile_and_run(_wgmma_gemm_prim_func, target_str, [-1], A, B)

    ref = A.float() @ B.float().t()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Test 2: two wgmma_gemm calls accumulating into one fragment (K split),
#         mirroring sparse_mla_fwd_pipelined's acc_s accumulation pattern
# ---------------------------------------------------------------------------

_ACC_M, _ACC_K, _ACC_N = 64, 128, 64
_ACC_KH = _ACC_K // 2


@T.prim_func
def _wgmma_gemm_accum_prim_func(
    A: T.Tensor((_M, _ACC_K), "float16"),
    B: T.Tensor((_ACC_N, _ACC_K), "float16"),
    C: T.Tensor((_ACC_M, _ACC_N), "float32"),
):
    with T.Kernel(1, threads=128):
        sa_l = T.alloc_shared([_ACC_M, _ACC_KH], T.float16)
        sa_r = T.alloc_shared([_ACC_M, _ACC_KH], T.float16)
        sb_l = T.alloc_shared([_ACC_N, _ACC_KH], T.float16)
        sb_r = T.alloc_shared([_ACC_N, _ACC_KH], T.float16)
        acc = T.alloc_fragment([_ACC_M, _ACC_N], T.float32)
        T.copy(A[0:_ACC_M, 0:_ACC_KH], sa_l)
        T.copy(A[0:_ACC_M, _ACC_KH:_ACC_K], sa_r)
        T.copy(B[0:_ACC_N, 0:_ACC_KH], sb_l)
        T.copy(B[0:_ACC_N, _ACC_KH:_ACC_K], sb_r)
        T.fill(acc, 0.0)
        T.wgmma_gemm(sa_l, sb_l, acc, transpose_B=True)
        T.wgmma_gemm(sa_r, sb_r, acc, transpose_B=True)
        T.wait_wgmma(0)
        T.copy(acc, C)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_wgmma_gemm_accumulate_numerical(monkeypatch):
    """Two wgmma_gemm K-split accumulations into one fragment equal A @ B^T."""
    _skip_if_tileir_toolchain_unavailable()
    torch, target_str = _setup_gpu()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    A = torch.randn(_ACC_M, _ACC_K, dtype=torch.float16, device="cuda")
    B = torch.randn(_ACC_N, _ACC_K, dtype=torch.float16, device="cuda")
    out = _compile_and_run(_wgmma_gemm_accum_prim_func, target_str, [-1], A, B)

    ref = A.float() @ B.float().t()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
