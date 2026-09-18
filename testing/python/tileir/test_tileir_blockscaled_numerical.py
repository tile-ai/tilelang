"""Numerical and MLIR-structural tests for the TileIR backend's scoped
recognition of ``T.tcgen05_gemm_blockscaled``.

The TileIR backend recognizes ``T.tcgen05_gemm_blockscaled`` calls (a
``tl.tileop.tcgen05_gemm_blockscaled`` carrying ``sf_a_granularity_k``/``sf_b_granularity_k``
annotations) within a narrow scope:

- ``k_start`` must be statically 0 (a single whole-K MMA; real hardware
  K-splits with ``k_start = k * block_K`` inside a loop).
- ``use_2cta`` is rejected (a distinct hardware path, unimplemented here).
- SFA/SFB must be plain e8m0/e4m3/uint8/int8 scale buffers, never the
  packed-uint32 TMEM layout the real examples use.
- SFA/SFB shapes must match ``(M, K // sf_a_granularity_k)`` /
  ``(K // sf_b_granularity_k, N)``, and K must be evenly divisible by both.

Anything outside this scope is rejected loudly
(``TileIRLoweringNotImplementedError``) rather than silently mis-lowered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

if TYPE_CHECKING:
    import torch

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


def _setup_sm100_gpu():
    """Like _setup_gpu, but skip below sm_100: mmaf_scaled is Blackwell-only,
    and the blockscaled lowering itself rejects earlier archs."""
    torch, major, minor, target_str = _setup_gpu()
    if (major, minor) < (10, 0):
        pytest.skip(f"blockscaled GEMM requires sm_100+; device is sm_{major}{minor}")
    return torch, major, minor, target_str


def _lower_first_kernel(pf):
    """Trace-level helper: run *pf* through the real semantic-extraction and
    sem_to_ir lowering pipeline (no cuda_tile / MLIR emission involved), and
    return nothing -- callers wrap this in ``pytest.raises`` to assert a
    lowering-time rejection. Used for the scoped-rejection tests below so
    they run unconditionally (no cuda_tile / GPU dependency)."""
    from tilelang.tileir.semantic import extract_semantic_program, materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    program = extract_semantic_program(pf)
    builder = IRBuilder()
    lower_kernel(program.kernels[0], builder, program=program)


# ---------------------------------------------------------------------------
# Test: SCOPED T.tcgen05_gemm_blockscaled recognition
# ---------------------------------------------------------------------------


def _tcgen05_blockscaled_gemm_jit(
    m: int = 64, n: int = 64, k: int = 64, v: int = 32, k_start: int = 0, op_name: str = "tcgen05_gemm_blockscaled"
):
    gemm = getattr(T, op_name)

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((k, n), "float8_e4m3"),
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, k // v), "uint8"),  # e8m0 bit patterns (torch 2.6 has no e8m0 dtype)
            SFB: T.Tensor((k // v, n), "uint8"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((k, n), "float8_e4m3")
                SFA_s = T.alloc_shared((m, k // v), "uint8")
                SFB_s = T.alloc_shared((k // v, n), "uint8")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                gemm(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    k_start=k_start,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                )
                T.copy(C_f, C)

        return main

    return make


@skip_no_cuda_tile
def test_tcgen05_gemm_blockscaled_emits_mmaf_scaled():
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _tcgen05_blockscaled_gemm_jit().get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))
    assert "mmaf_scaled" in mlir


@skip_no_cuda_tile
def test_tcgen05_gemm_blockscaled_rejects_pre_sm100():
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _tcgen05_blockscaled_gemm_jit().get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    with pytest.raises(TileIRLoweringError, match="sm_100"):
        build_tileir_module(pf, arch="sm_90")


def test_tcgen05_gemm_blockscaled_semantic_regions():
    from tilelang.tileir.semantic import extract_semantic_program, materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc
    from tileir_test_utils import _semantic_stmts

    pf = _tcgen05_blockscaled_gemm_jit().get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    program = extract_semantic_program(pf)

    gemm_stmts = [
        s
        for k in program.kernels
        for s in _semantic_stmts(k.body)
        if s.kind == "tile_op" and dict(s.attrs).get("op") == "tl.tileop.tcgen05_gemm_blockscaled"
    ]
    assert len(gemm_stmts) == 1
    stmt = gemm_stmts[0]
    assert len(stmt.regions) == 5, f"expected 5 regions (A,B,C,SFA,SFB), got {len(stmt.regions)}"
    attrs = dict(stmt.attrs)
    assert attrs.get("annotation.sf_a_granularity_k") == "32"
    assert attrs.get("annotation.sf_b_granularity_k") == "32"
    assert attrs.get("k_start") == "0"


def _e8m0_to_float(bits: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.pow(2.0, bits.to(torch.float32) - 127.0)


@skip_no_cuda_tile
def test_tcgen05_gemm_blockscaled_mxfp8_identity_scale_numerical():
    """Identity scale (all e8m0 bits == 127 -> scale factor 1.0) isolates
    fp8 rounding from scale application: this should match a plain fp8
    matmul reference to a tight tolerance. If this passes but the
    random-exponent variant below fails, the bug is in scale application
    (wrong operand order, missing bitcast, transposed scale tile) rather
    than in the base MMA path."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_sm100_gpu()
    m = n = k = 64
    v = 32

    kernel = tilelang.compile(_tcgen05_blockscaled_gemm_jit(m, n, k, v).get_tir(), execution_backend="tileir")

    a = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(k, n, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    sfa = torch.full((m, k // v), 127, dtype=torch.uint8, device="cuda")
    sfb = torch.full((k // v, n), 127, dtype=torch.uint8, device="cuda")
    c = torch.empty(m, n, dtype=torch.float32, device="cuda")
    kernel(a, b, c, sfa, sfb)

    ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, ref, rtol=5e-2, atol=5e-2)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_name", ["tcgen05_gemm_blockscaled", "gemm_blockscaled"])
def test_tcgen05_gemm_blockscaled_mxfp8_numerical(op_name):
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_sm100_gpu()
    m = n = k = 64
    v = 32

    kernel = tilelang.compile(_tcgen05_blockscaled_gemm_jit(m, n, k, v, op_name=op_name).get_tir(), execution_backend="tileir")

    a = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(k, n, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    # e8m0 bit patterns near 1.0: randint(125, 130) -> exponents -2..2 (bias
    # 127) -> scales 0.25..4
    sfa = torch.randint(125, 130, (m, k // v), dtype=torch.uint8, device="cuda")
    sfb = torch.randint(125, 130, (k // v, n), dtype=torch.uint8, device="cuda")
    c = torch.empty(m, n, dtype=torch.float32, device="cuda")
    kernel(a, b, c, sfa, sfb)

    a_deq = a.to(torch.float32) * _e8m0_to_float(sfa).repeat_interleave(v, dim=1)
    b_deq = b.to(torch.float32) * _e8m0_to_float(sfb).repeat_interleave(v, dim=0)
    ref = a_deq @ b_deq
    torch.testing.assert_close(c, ref, rtol=5e-2, atol=5e-1)


def _tcgen05_blockscaled_gemm_transpose_b_jit(m: int = 64, n: int = 64, k: int = 64, v: int = 32):
    """B is stored (n, k) fp8 (transpose_B=True); SFB is still declared and
    validated against the LOGICAL (k // v, n) layout regardless of
    transpose_B -- the emit path (compute.py's _emit_gemm_scaled_impl,
    docstring point 5) never permutes the scale tiles: only LHS/RHS are
    permuted for trans_a/trans_b, so the caller must already allocate SFB in
    logical (post-transpose) orientation."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((n, k), "float8_e4m3"),  # stored (n, k): transpose_B=True
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, k // v), "uint8"),
            SFB: T.Tensor((k // v, n), "uint8"),  # logical (k // v, n), NOT (n // v, k)
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((n, k), "float8_e4m3")
                SFA_s = T.alloc_shared((m, k // v), "uint8")
                SFB_s = T.alloc_shared((k // v, n), "uint8")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                T.tcgen05_gemm_blockscaled(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    transpose_B=True,
                    k_start=0,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                )
                T.copy(C_f, C)

        return main

    return make


@skip_no_cuda_tile
def test_tcgen05_gemm_blockscaled_mxfp8_transpose_b_numerical():
    """Transposed-B numeric proof: B is stored (n, k) and dequantized against
    SFB declared in the logical (k // v, n) orientation."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_sm100_gpu()
    m = n = k = 64
    v = 32

    kernel = tilelang.compile(_tcgen05_blockscaled_gemm_transpose_b_jit(m, n, k, v).get_tir(), execution_backend="tileir")

    a = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b_stored = (torch.randn(n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    # e8m0 bit patterns near 1.0: randint(125, 130) -> exponents -2..2 (bias
    # 127) -> scales 0.25..4
    sfa = torch.randint(125, 130, (m, k // v), dtype=torch.uint8, device="cuda")
    sfb = torch.randint(125, 130, (k // v, n), dtype=torch.uint8, device="cuda")
    c = torch.empty(m, n, dtype=torch.float32, device="cuda")
    kernel(a, b_stored, c, sfa, sfb)

    b_logical = b_stored.t()  # (n, k) -> (k, n)
    a_deq = a.to(torch.float32) * _e8m0_to_float(sfa).repeat_interleave(v, dim=1)
    b_deq = b_logical.to(torch.float32) * _e8m0_to_float(sfb).repeat_interleave(v, dim=0)
    ref = a_deq @ b_deq
    torch.testing.assert_close(c, ref, rtol=5e-2, atol=5e-1)


# ---------------------------------------------------------------------------
# Test: scoped rejections (no GPU / no cuda_tile needed -- these exercise the
# pure-Python sem_to_ir lowering layer, tile_ops.py's _lower_gemm_scaled)
# ---------------------------------------------------------------------------


def test_tcgen05_gemm_blockscaled_rejects_k_start_nonzero():
    """K-split usage (k_start != 0) is the form every real hardware example
    uses, and is explicitly out of scope for this backend (see module
    docstring) -- it must be rejected loudly, not silently mis-lowered as an
    unscaled or wrongly-scaled MMA."""
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    pf = _tcgen05_blockscaled_gemm_jit(k_start=64).get_tir()
    with pytest.raises(TileIRLoweringNotImplementedError, match="k_start"):
        _lower_first_kernel(pf)


def _tcgen05_blockscaled_gemm_2cta_jit(m: int = 64, n: int = 64, k: int = 64, v: int = 32):
    """use_2cta=True: B/SFB hold half of N per CTA (N_B = N // 2), matching
    the frontend's 2CTA shape assertion in gemm_op.py."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((k, n // 2), "float8_e4m3"),
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, k // v), "uint8"),
            SFB: T.Tensor((k // v, n // 2), "uint8"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((k, n // 2), "float8_e4m3")
                SFA_s = T.alloc_shared((m, k // v), "uint8")
                SFB_s = T.alloc_shared((k // v, n // 2), "uint8")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                T.tcgen05_gemm_blockscaled(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                    use_2cta=True,
                )
                T.copy(C_f, C)

        return main

    return make


def test_tcgen05_gemm_blockscaled_rejects_use_2cta():
    """The true 2CTA lowering is a distinct hardware path this backend does
    not implement; it must be rejected loudly rather than silently lowered
    as a (wrong) single-CTA MMA."""
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    pf = _tcgen05_blockscaled_gemm_2cta_jit().get_tir()
    with pytest.raises(TileIRLoweringNotImplementedError, match="use_2cta"):
        _lower_first_kernel(pf)


def _tcgen05_blockscaled_gemm_jit_bad_scale_dtype(m: int = 64, n: int = 64, k: int = 64, v: int = 32):
    """SFA/SFB declared uint32 -- the packed-4-scales-per-word TMEM layout
    the real tcgen05 hardware examples use, which this backend does not
    reverse-engineer (see module docstring)."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((k, n), "float8_e4m3"),
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, k // v), "uint32"),
            SFB: T.Tensor((k // v, n), "uint32"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((k, n), "float8_e4m3")
                SFA_s = T.alloc_shared((m, k // v), "uint32")
                SFB_s = T.alloc_shared((k // v, n), "uint32")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                T.tcgen05_gemm_blockscaled(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                )
                T.copy(C_f, C)

        return main

    return make


def test_tcgen05_gemm_blockscaled_rejects_packed_scale_dtype():
    """uint32-packed scale buffers (the real tcgen05 TMEM layout) must be
    rejected loudly rather than silently misinterpreted."""
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    pf = _tcgen05_blockscaled_gemm_jit_bad_scale_dtype().get_tir()
    with pytest.raises(TileIRLoweringNotImplementedError, match="packed"):
        _lower_first_kernel(pf)


def _tcgen05_blockscaled_gemm_jit_bad_sfa_shape(m: int = 64, n: int = 64, k: int = 64, v: int = 32):
    """SFA declared with a shape that does not match (M, K // sf_a_granularity_k).
    The frontend (T.tcgen05_gemm_blockscaled) does no shape validation of its
    own -- this must be caught by the lowering layer."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((k, n), "float8_e4m3"),
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, 3), "uint8"),  # wrong: should be (m, k // v)
            SFB: T.Tensor((k // v, n), "uint8"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((k, n), "float8_e4m3")
                SFA_s = T.alloc_shared((m, 3), "uint8")
                SFB_s = T.alloc_shared((k // v, n), "uint8")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                T.tcgen05_gemm_blockscaled(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                )
                T.copy(C_f, C)

        return main

    return make


def test_tcgen05_gemm_blockscaled_rejects_wrong_sfa_shape():
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    pf = _tcgen05_blockscaled_gemm_jit_bad_sfa_shape().get_tir()
    with pytest.raises(TileIRLoweringNotImplementedError, match="SFA"):
        _lower_first_kernel(pf)


def _tcgen05_blockscaled_gemm_jit_bad_granularity(m: int = 64, n: int = 64, k: int = 48, v: int = 32):
    """K=48 is not evenly divisible by v=32 (floor(K / V) == 1 is used for
    the SFA/SFB shapes so the granularity-divisibility check -- not the
    shape-match check -- is what fires)."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float8_e4m3"),
            B: T.Tensor((k, n), "float8_e4m3"),
            C: T.Tensor((m, n), "float32"),
            SFA: T.Tensor((m, k // v), "uint8"),
            SFB: T.Tensor((k // v, n), "uint8"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, k), "float8_e4m3")
                B_s = T.alloc_shared((k, n), "float8_e4m3")
                SFA_s = T.alloc_shared((m, k // v), "uint8")
                SFB_s = T.alloc_shared((k // v, n), "uint8")
                C_f = T.alloc_fragment((m, n), "float32")
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.copy(SFA, SFA_s)
                T.copy(SFB, SFB_s)
                T.tcgen05_gemm_blockscaled(
                    A_s,
                    B_s,
                    C_f,
                    SFA_s,
                    SFB_s,
                    mbar=bar[0],
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=v,
                    sf_b_granularity_k=v,
                )
                T.copy(C_f, C)

        return main

    return make


def test_tcgen05_gemm_blockscaled_rejects_non_divisible_granularity():
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    pf = _tcgen05_blockscaled_gemm_jit_bad_granularity().get_tir()
    with pytest.raises(TileIRLoweringNotImplementedError, match="divisible"):
        _lower_first_kernel(pf)


# ---------------------------------------------------------------------------
# Test: lowering-layer defensive guard (no semantic layer validation)
# ---------------------------------------------------------------------------


def test_lower_gemm_blockscaled_guards_region_count():
    """The lowering layer must reject annotation-without-5-regions itself
    (defensive contract; the semantic layer normally enforces this)."""
    import pytest as _pytest
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt, SemanticProgram, SemanticRegion
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    # Construct a blockscaled gemm stmt with only 3 regions (missing SFA/SFB).
    lhs_buf = SemanticBuffer(name="lhs", shape=(32, 64), dtype="float8_e4m3", scope="shared")
    rhs_buf = SemanticBuffer(name="rhs", shape=(64, 32), dtype="float8_e4m3", scope="shared")
    acc_buf = SemanticBuffer(name="acc", shape=(32, 32), dtype="float32", scope="local")

    gemm_stmt = SemanticStmt(
        kind="tile_op",
        attrs=(
            ("op", "tl.tileop.gemm"),
            ("annotation.sf_a_granularity_k", "32"),
            ("annotation.sf_b_granularity_k", "32"),
            ("transpose_A", "0"),
            ("transpose_B", "0"),
            ("clear_accum", "1"),
        ),
        regions=(
            SemanticRegion(buffer="lhs", access="read", indices=(), shape=(32, 64)),
            SemanticRegion(buffer="rhs", access="read", indices=(), shape=(64, 32)),
            SemanticRegion(buffer="acc", access="readwrite", indices=(), shape=(32, 32)),
        ),
    )
    body = SemanticStmt(kind="seq", children=(gemm_stmt,))
    kernel = SemanticKernel(
        name="blockscaled_guard_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(lhs_buf, rhs_buf, acc_buf),
        body=body,
    )
    program = SemanticProgram(
        name="blockscaled_guard_test_prog",
        params=(),
        global_alloc_buffers=(),
        kernels=(kernel,),
    )

    # Attempt to lower the kernel; expect TileIRLoweringError about 5 regions.
    builder = IRBuilder()
    with _pytest.raises(TileIRLoweringError, match="5 regions"):
        lower_kernel(kernel, builder, program=program)


# ---------------------------------------------------------------------------
# Test: runtime dtype mapping (no GPU needed)
# ---------------------------------------------------------------------------


def test_runtime_dtype_map_fp8():
    from tilelang.jit.adapter.tileir.runtime import torch_dtype_from_tileir
    import torch as _torch

    assert torch_dtype_from_tileir("float8_e4m3fn") == _torch.float8_e4m3fn
    assert torch_dtype_from_tileir("float8_e5m2") == _torch.float8_e5m2
    assert torch_dtype_from_tileir("float8_e8m0fnu") == _torch.uint8
