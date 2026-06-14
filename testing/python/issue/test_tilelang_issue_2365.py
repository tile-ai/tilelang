"""Regression test for issue #2365.

A pipelined fp16/bf16/fp8 GEMM whose K leaves a 1-element residual tile
(K = BLOCK_K + 1, e.g. K=33 with BLOCK_K=32) used to crash at CUDA codegen with
"tl::ptx_cp_async requires a final PTX byte width in {4, 8, 16}, but got 2": the
residual copy's vec-dependent predicate forces LegalizeVectorizedLoop to
scalarize the vectorized cp.async into a num_elems==1 (sub-16-byte) transfer,
which is not a legal cp.async width. The fix demotes such copies to a
synchronous masked store with a typed zero-fill (matching cp.async's hardware
zero-fill on a false predicate).
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch

# cp.async exists from sm_80 (Ampere). fp8 e4m3 GEMM execution wants sm_89+.
_MIN_CC = (8, 0)

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3": torch.float8_e4m3fn,
    "float32": torch.float32,
}
# (rtol, atol) per dtype, matching the validated trigger surface.
_TOL = {
    "float16": (2e-2, 2e-1),
    "bfloat16": (6e-2, 5e-1),
    "float8_e4m3": (2e-1, 2.0),
    "float32": (2e-2, 2e-1),
}


def _make_gemm(M, N, K, block_M, block_N, block_K, dtype, num_stages=2):
    """Build a pipelined C = A @ B kernel (fp32 accum) for the given shapes/dtype."""
    accum = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum),
    ):
        """Pipelined tiled GEMM over K with cp.async global->shared copies."""
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            As = T.alloc_shared((block_M, block_K), dtype)
            Bs = T.alloc_shared((block_K, block_N), dtype)
            Cl = T.alloc_fragment((block_M, block_N), accum)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], As)
                T.copy(B[k * block_K, bx * block_N], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * block_M, bx * block_N])

    return main


def _rand(shape, dtype):
    """Random CUDA tensor of the given tilelang dtype (scaled down for fp8)."""
    if dtype == "float8_e4m3":
        # Small magnitudes so e4m3 can represent them and the fp32 accumulation is sane.
        return (torch.randn(*shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    return torch.randn(*shape, device="cuda", dtype=_TORCH_DTYPE[dtype])


def _compile_run(M, N, K, block_M, block_N, block_K, dtype, num_stages=2):
    """Compile + run; returns (output_fp32, reference_fp32). Compilation errors
    propagate (that IS the detection of the original crash)."""
    func = _make_gemm(M, N, K, block_M, block_N, block_K, dtype, num_stages)
    kernel = tilelang.compile(func, out_idx=[2], execution_backend="cython")
    a = _rand((M, K), dtype)
    b = _rand((K, N), dtype)
    c = kernel(a, b).float()
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    return c, ref


# (name, M, N, K, block_M, block_N, block_K, dtype) -- residual = 1 element, the crash trigger.
_TRIGGER_CASES = [
    ("fp16 K33/BK32 (headline)", 128, 128, 33, 64, 64, 32, "float16"),
    ("fp16 K65/BK64", 128, 128, 65, 64, 64, 64, "float16"),
    ("fp16 K17/BK16", 128, 128, 17, 64, 64, 16, "float16"),
    ("fp16 K97/BK96", 128, 128, 97, 64, 64, 96, "float16"),
    ("fp16 K129/BK128", 128, 128, 129, 64, 64, 128, "float16"),
    ("bf16 K33/BK32", 128, 128, 33, 64, 64, 32, "bfloat16"),
    ("bf16 K17/BK16", 128, 128, 17, 64, 64, 16, "bfloat16"),
]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(*_MIN_CC)
def test_pipelined_residual_gemm_compiles_and_correct():
    """The original codegen crash: each trigger case must compile, run, produce
    no NaN, and match torch."""
    failures = []
    for name, M, N, K, bM, bN, bK, dt in _TRIGGER_CASES:
        rtol, atol = _TOL[dt]
        try:
            c, ref = _compile_run(M, N, K, bM, bN, bK, dt)
        except RuntimeError as e:  # the bug we guard: tvm InternalError <- RuntimeError
            failures.append(f"{name}: COMPILE/RUN RAISED: {str(e).splitlines()[-1][:120]}")
            continue
        nan = int(torch.isnan(c).sum().item())
        if nan:
            failures.append(f"{name}: output has {nan} NaNs (missing zero-fill?)")
        elif not torch.allclose(c, ref, rtol=rtol, atol=atol):
            max_abs = (c - ref).abs().max().item()
            failures.append(f"{name}: numeric mismatch max_abs={max_abs:.4g}")
    assert not failures, "illegal-width residual cp.async regressions:\n  " + "\n  ".join(failures)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(*_MIN_CC)
def test_zero_fill_residual_no_garbage():
    """The synchronous fallback must zero-fill out-of-bounds residual lanes.

    Poison shared memory with NaN via a full-tile (no-residual) GEMM, then run
    the 1-element-residual GEMM in the same process so it reuses the same SM
    shared-memory region. If the residual's out-of-bounds lanes are not
    zero-filled, they retain NaN and the result is NaN (deterministic).
    """
    for dt in ("float16", "bfloat16"):
        rtol, atol = _TOL[dt]
        block_M = block_N = 64
        block_K = 32
        poisoner = tilelang.compile(
            _make_gemm(128, 128, block_K, block_M, block_N, block_K, dt),  # K == BK: full tile, no residual
            out_idx=[2],
            execution_backend="cython",
        )
        residual = tilelang.compile(
            _make_gemm(128, 128, block_K + 1, block_M, block_N, block_K, dt),  # K = BK+1: 1-elem residual
            out_idx=[2],
            execution_backend="cython",
        )
        nan_runs = 0
        for _ in range(8):
            pa = torch.full((128, block_K), float("nan"), device="cuda", dtype=_TORCH_DTYPE[dt])
            pb = torch.randn(block_K, 128, device="cuda", dtype=_TORCH_DTYPE[dt])
            poisoner(pa, pb)
            torch.cuda.synchronize()

            a = torch.randn(128, block_K + 1, device="cuda", dtype=_TORCH_DTYPE[dt])
            b = torch.randn(block_K + 1, 128, device="cuda", dtype=_TORCH_DTYPE[dt])
            c = residual(a, b).float()
            torch.cuda.synchronize()
            ref = a.float() @ b.float()
            if torch.isnan(c).any():
                nan_runs += 1
            else:
                assert torch.allclose(c, ref, rtol=rtol, atol=atol), (
                    f"{dt}: residual GEMM numerically wrong after poison (max_abs={(c - ref).abs().max().item():.4g})"
                )
        assert nan_runs == 0, (
            f"{dt}: {nan_runs}/8 residual runs produced NaN -- the synchronous "
            f"fallback is not zero-filling out-of-bounds shared-memory lanes "
            f"(cp.async zero-fill semantics not replicated)."
        )


# (name, M, N, K, block_M, block_N, block_K, dtype): never hit the illegal-width path; must stay correct.
_CONTROL_CASES = [
    ("fp16 K32 divisible", 128, 128, 32, 64, 64, 32, "float16"),
    ("fp16 K64 divisible", 128, 128, 64, 64, 64, 32, "float16"),
    ("fp16 K128 multi-tile", 128, 128, 128, 64, 64, 32, "float16"),
    ("fp16 K35 resid=3 (vec-indep, was-legal)", 128, 128, 35, 64, 64, 32, "float16"),
    ("fp16 K34 resid=2 (4B, legal)", 128, 128, 34, 64, 64, 32, "float16"),
    ("fp32 K33 resid=1 -> 4B (was-legal)", 128, 128, 33, 64, 64, 32, "float32"),
]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(*_MIN_CC)
def test_controls_unaffected():
    """Divisible-K and already-legal-width residual cases must remain correct
    (the fix must not demote copies that are valid as cp.async)."""
    failures = []
    for name, M, N, K, bM, bN, bK, dt in _CONTROL_CASES:
        rtol, atol = _TOL[dt]
        c, ref = _compile_run(M, N, K, bM, bN, bK, dt)
        if torch.isnan(c).any():
            failures.append(f"{name}: NaN in a control case")
        elif not torch.allclose(c, ref, rtol=rtol, atol=atol):
            failures.append(f"{name}: max_abs={(c - ref).abs().max().item():.4g}")
    assert not failures, "control-case regressions:\n  " + "\n  ".join(failures)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(*_MIN_CC)
def test_legal_fast_path_preserves_cp_async():
    """A divisible-K pipelined GEMM must still emit cp.async on its legal-width
    fast path: the fix must demote only the illegal residual copy, not the bulk
    copies."""
    func = _make_gemm(128, 128, 128, 64, 64, 32, "float16", num_stages=2)
    kernel = tilelang.compile(func, out_idx=[2], execution_backend="cython")
    src = kernel.get_kernel_source()
    assert ("cp_async_gs" in src) or ("cp.async" in src), (
        "divisible-K fast path no longer emits cp.async -- the fix appears to "
        "have demoted legal-width copies to synchronous (perf regression):\n" + src[:2000]
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)  # fp8 e4m3 GEMM execution
def test_pipelined_residual_gemm_fp8():
    """fp8 e4m3 residual = 1 byte (also illegal). Same crash class; gated to
    sm_89+ where fp8 GEMM actually executes."""
    rtol, atol = _TOL["float8_e4m3"]
    for name, K, bK in [("fp8 K33/BK32", 33, 32), ("fp8 K17/BK16", 17, 16)]:
        c, ref = _compile_run(128, 128, K, 64, 64, bK, "float8_e4m3")
        assert not torch.isnan(c).any(), f"{name}: NaN (missing zero-fill?)"
        assert torch.allclose(c, ref, rtol=rtol, atol=atol), f"{name}: max_abs={(c - ref).abs().max().item():.4g}"


if __name__ == "__main__":
    tilelang.testing.main()
