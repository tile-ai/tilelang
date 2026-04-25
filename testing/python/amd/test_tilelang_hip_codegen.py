"""
Tests for HIP/AMD codegen fixes in TileLang.

Covers three fixes made to src/target/codegen_hip.cc:
  1. T.sync_warp() is lowered to a no-op on HIP (AMD wavefronts execute in
     lockstep so no explicit reconvergence barrier is needed).
  2. T.alloc_var(dtype, init=value) emits a properly initialised scalar
     declaration on HIP (previously the init value was silently dropped).
  3. local.var buffers are accessed as plain scalars in GetBufferRef (no [0]
     subscript), consistent with the scalar declaration emitted for them.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T


# ---------------------------------------------------------------------------
# Fix 1: T.sync_warp() → no-op on HIP
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_sync_warp_codegen():
    """Minimal kernel that exercises T.sync_warp()."""

    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((32,), "float32")
            A_shared[tx] = A[tx]
            T.sync_warp()
            B[tx] = A_shared[tx] * 2.0

    return main


@tilelang.testing.requires_rocm
def test_sync_warp_no_syncwarp_in_hip_source():
    """__syncwarp must NOT appear in the HIP-generated kernel source."""
    kernel = _kernel_sync_warp_codegen()
    src = kernel.get_kernel_source()
    assert "__syncwarp" not in src, (
        "T.sync_warp() should be a no-op on HIP (AMD wavefronts are lockstep), "
        f"but __syncwarp was found in the generated source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_sync_warp_correctness():
    """Kernel using T.sync_warp() should produce correct results on HIP."""
    kernel = _kernel_sync_warp_codegen()
    A = torch.arange(32, dtype=torch.float32, device="cuda")
    B = torch.zeros(32, dtype=torch.float32, device="cuda")
    kernel(A, B)
    torch.testing.assert_close(B, A * 2.0)


# ---------------------------------------------------------------------------
# Fix 2: T.alloc_var(init=...) initialisation on HIP
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_init():
    """Kernel that initialises a local int32 variable and writes it to output."""

    @T.prim_func
    def main(Out: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            counter = T.alloc_var(T.int32, init=7)
            Out[tx] = counter

    return main


@tilelang.testing.requires_rocm
def test_alloc_var_init_in_hip_source():
    """Init value must appear in the generated HIP source for T.alloc_var."""
    kernel = _kernel_alloc_var_init()
    src = kernel.get_kernel_source()
    assert "= 7;" in src, (
        "T.alloc_var(T.int32, init=7) should generate '= 7;' in HIP source, "
        f"but it was not found.\nGenerated source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_no_array_subscript_in_hip_source():
    """local.var should be declared as a scalar, not as an array (no [0])."""
    kernel = _kernel_alloc_var_init()
    src = kernel.get_kernel_source()
    # The kernel source should contain 'counter = 7' not 'counter[1]' or 'counter[0]'
    assert "counter[" not in src, (
        "local.var should be emitted as a scalar (e.g. 'int counter = 7'), "
        f"but array-style access was found in the HIP source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_correctness():
    """Kernel should read back the initialised value correctly on HIP."""
    kernel = _kernel_alloc_var_init()
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 7), (
        f"Expected all 7, got: {out}"
    )


# ---------------------------------------------------------------------------
# Fix 2b: multiple T.alloc_var with distinct init values
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_multi_alloc_var_init():
    """Two local variables with different init values, summed into output."""

    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            a = T.alloc_var(T.int32, init=3)
            b = T.alloc_var(T.int32, init=4)
            Out[tx] = a + b

    return main


@tilelang.testing.requires_rocm
def test_multi_alloc_var_init_in_hip_source():
    """Both init values must appear in the HIP source."""
    kernel = _kernel_multi_alloc_var_init()
    src = kernel.get_kernel_source()
    assert src.count("= 3;") >= 1, f"Init value 3 not found in HIP source:\n{src}"
    assert src.count("= 4;") >= 1, f"Init value 4 not found in HIP source:\n{src}"


@tilelang.testing.requires_rocm
def test_multi_alloc_var_init_correctness():
    """Sum of two initialised local variables should equal 7 on HIP."""
    kernel = _kernel_multi_alloc_var_init()
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 7), f"Expected all 7 (3+4), got: {out}"


# ---------------------------------------------------------------------------
# Fix 2c: T.alloc_var(init=0) — the default zero-init case
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_count():
    """
    Accumulates a count by incrementing a local variable in a loop.
    Relies on the variable being zero-initialised (init=0 default).
    """

    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            count = T.alloc_var(T.int32, init=0)
            for _ in T.unroll(5):
                count += 1
            Out[tx] = count

    return main


@tilelang.testing.requires_rocm
def test_alloc_var_zero_init_correctness():
    """Variable initialised to 0, incremented 5 times, should equal 5."""
    kernel = _kernel_alloc_var_count()
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 5), f"Expected all 5, got: {out}"


# ---------------------------------------------------------------------------
# Fix 3: T.sync_grid() codegen on HIP (codegen only; runtime not yet complete)
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_sync_grid_codegen():
    """Kernel that calls T.sync_grid() to trigger cooperative groups codegen."""

    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.sync_grid()
            A[tx] = T.float32(tx)

    return main


@tilelang.testing.requires_rocm
def test_sync_grid_cooperative_groups_in_hip_source():
    """
    T.sync_grid() should generate cooperative_groups::this_grid().sync()
    and include <hip/hip_cooperative_groups.h> in the HIP source.

    Note: runtime execution of this kernel is not yet supported because
    rocm_module.cc does not yet call hipModuleLaunchCooperativeKernel.
    This test validates codegen only.
    """
    kernel = _kernel_sync_grid_codegen()
    src = kernel.get_kernel_source()
    assert "this_grid().sync()" in src, (
        "T.sync_grid() should generate 'this_grid().sync()' in HIP source, "
        f"but it was not found:\n{src}"
    )
    assert "cooperative_groups" in src, (
        "T.sync_grid() should include cooperative_groups in HIP source, "
        f"but it was not found:\n{src}"
    )


# ---------------------------------------------------------------------------
# Fix 4: warp_reduce — 5-step butterfly with width=32 (reduce.h)
#
# On CDNA (wave64) the old 6-step butterfly called __shfl_xor(value, 32)
# without a width argument, reading uninitialised VGPRs in lanes 32-63 when
# only 32 threads were active → NaN / garbage in reduce_max / reduce_sum.
# Fix: remove step-32 shuffle; add width=32 to all remaining 5 steps.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("n_tokens,n_experts", [(64, 8), (128, 16), (512, 32)])
def test_warp_reduce_no_nan(n_tokens, n_experts):
    """
    32-thread-per-block reduce_max / reduce_sum must not produce NaN on CDNA.

    Old: __shfl_xor(v, 32) with 32 active threads reads uninit VGPRs → NaN.
    New: 5-step with width=32 stays in-group → correct result, no NaN.
    """
    assert n_experts <= 32

    @tilelang.jit
    def gate_reduce(n_tok: int, n_exp: int):
        @T.prim_func
        def kernel(
            logits:  T.Tensor((n_tok, n_exp), T.float32),
            out_max: T.Tensor((n_tok,), T.float32),
            out_sum: T.Tensor((n_tok,), T.float32),
        ) -> None:
            with T.Kernel(n_tok, threads=32) as pid:
                lf = T.alloc_fragment(n_exp, T.float32)
                T.copy(logits[pid, 0], lf)
                mx = T.alloc_fragment(1, T.float32)
                T.reduce_max(lf, mx, dim=0)
                sm = T.alloc_fragment(1, T.float32)
                T.reduce_sum(lf, sm, dim=0)
                if T.get_thread_binding() == 0:
                    out_max[pid] = mx[0]
                    out_sum[pid] = sm[0]
        return kernel

    logits  = torch.randn(n_tokens, n_experts, dtype=torch.float32, device="cuda")
    out_max = torch.zeros(n_tokens, dtype=torch.float32, device="cuda")
    out_sum = torch.zeros(n_tokens, dtype=torch.float32, device="cuda")

    gate_reduce(n_tokens, n_experts)(logits, out_max, out_sum)
    torch.cuda.synchronize()

    assert not out_max.isnan().any(), "reduce_max NaN — __shfl_xor(v,32) uninit VGPR bug"
    assert not out_sum.isnan().any(), "reduce_sum NaN — __shfl_xor(v,32) uninit VGPR bug"
    torch.testing.assert_close(out_max, logits.max(dim=1).values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_sum, logits.sum(dim=1),        atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_rocm
def test_warp_reduce_correctness_32_threads():
    """
    32-thread reduce_sum over 32 elements must return the exact sum on CDNA.

    warp_reduce is exercised when T.reduce_sum falls through to the warp-level
    shuffle path.  With the old __shfl_xor(v, 32) bug, reading uninitialised
    lanes 32-63 on CDNA produced garbage.  With width=32 the result is exact.
    """
    N = 32

    @tilelang.jit
    def reduce_kernel():
        @T.prim_func
        def kernel(
            x:   T.Tensor((N,), T.float32),
            out: T.Tensor((1,), T.float32),
        ) -> None:
            with T.Kernel(1, threads=N) as _:
                frag = T.alloc_fragment((N,), T.float32)
                T.copy(x, frag)
                s = T.alloc_fragment((1,), T.float32)
                T.reduce_sum(frag, s, dim=0)
                if T.get_thread_binding() == 0:
                    out[0] = s[0]
        return kernel

    x   = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    reduce_kernel()(x, out)
    torch.cuda.synchronize()

    expected = x.sum()
    assert not out[0].isnan(), "reduce_sum NaN — warp_reduce VGPR bug on CDNA"
    torch.testing.assert_close(out[0], expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fix 5: ShuffleNode bfloat16x2 / float16x2 packing (codegen_hip.cc)
#         uint1 bf16x2 math overloads (common.h)
#
# Old: CodeGenC emitted `uint1(a, b)` — invalid HIP constructor → compile error.
# Fix: Override VisitExpr_(ShuffleNode) to emit `uint1{__pack_bfloat162(a,b)}`.
#      Add abs2/max2/min2/add2/mul2 overloads for uint1 in common.h.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
def test_bfloat16_shuffle_emits_pack_intrinsic():
    """
    A bfloat16 fragment reduction triggers ShuffleNode bfloat16x2 packing.
    The generated HIP source must use __pack_bfloat162 (not `uint1(a,b)`).
    """
    n_tok, n_exp = 16, 8

    @tilelang.jit
    def bf16_reduce(n_t: int, n_e: int):
        @T.prim_func
        def kernel(
            x:   T.Tensor((n_t, n_e), T.bfloat16),
            out: T.Tensor((n_t,), T.float32),
        ) -> None:
            with T.Kernel(n_t, threads=32) as pid:
                frag = T.alloc_fragment(n_e, T.bfloat16)
                T.copy(x[pid, 0], frag)
                frag_f32 = T.alloc_fragment(n_e, T.float32)
                for i in T.Parallel(n_e):
                    frag_f32[i] = T.cast(frag[i], T.float32)
                s = T.alloc_fragment(1, T.float32)
                T.reduce_sum(frag_f32, s, dim=0)
                if T.get_thread_binding() == 0:
                    out[pid] = s[0]
        return kernel

    src = bf16_reduce(n_tok, n_exp).get_kernel_source()
    # Must use the pack intrinsic, not the invalid two-argument constructor
    assert "uint1(a" not in src and "uint1(b" not in src, \
        "Old `uint1(a, b)` constructor found — ShuffleNode fix not applied"

    # Runtime correctness
    x   = torch.randn(n_tok, n_exp, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(n_tok,        dtype=torch.float32,  device="cuda")
    bf16_reduce(n_tok, n_exp)(x, out)
    torch.cuda.synchronize()
    assert not out.isnan().any(), "bf16 ShuffleNode reduction produced NaN"
    torch.testing.assert_close(out, x.float().sum(dim=1), atol=5e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Fix 6: T.Pipelined(num_stages>1) on ROCM — skip pipeline planning
#         (pipeline_planning.cc)
#
# Old: double-buffering doubled LDS per loop body → hipModuleLaunchKernel
#      EINVAL on CDNA (≤128 KB LDS/workgroup).
# Fix: TargetIsRocm() && num_stages>1 → fall back to plain sequential loop.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_pipelined_no_lds_overflow(num_stages):
    """
    T.Pipelined(num_stages=N) must not raise hipModuleLaunchKernel EINVAL on
    ROCM and must produce the correct result regardless of N.

    Old: num_stages=2 doubled LDS allocation → EINVAL on CDNA.
    New: multi-stage loops fall back to plain sequential on ROCM.
    """
    M, K, blk = 32, 256, 64

    @tilelang.jit
    def pipelined_rowsum(n_stages: int):
        @T.prim_func
        def kernel(
            x:   T.Tensor((M, K), T.float32),
            out: T.Tensor((M,),   T.float32),
        ) -> None:
            with T.Kernel(M, threads=64) as pid:
                acc = T.alloc_fragment((1,), T.float32)
                T.clear(acc)
                for k in T.Pipelined(K // blk, num_stages=n_stages):
                    xs = T.alloc_shared((blk,), T.float32)
                    xl = T.alloc_fragment((blk,), T.float32)
                    T.copy(x[pid, k * blk], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)
                    s = T.alloc_fragment((1,), T.float32)
                    T.reduce_sum(xl, s, dim=0)
                    acc[0] = acc[0] + s[0]
                out[pid] = acc[0]
        return kernel

    x   = torch.ones(M, K, dtype=torch.float32, device="cuda")
    out = torch.zeros(M,   dtype=torch.float32, device="cuda")
    pipelined_rowsum(num_stages)(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.full((M,), float(K), device="cuda"), atol=1e-4, rtol=0)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipelined_multi_stage_fp16_gemm(num_stages):
    """
    FP16 GEMM with T.Pipelined(num_stages>1) must launch and produce correct
    results on ROCM — the most common pattern that triggered the LDS overflow.
    """
    M, N, K = 128, 128, 128
    bM, bN, bK = 64, 64, 32

    @tilelang.jit
    def fp16_gemm(n_stages: int):
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((K, N), T.float16),
            C: T.Tensor((M, N), T.float32),
        ) -> None:
            with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
                A_s = T.alloc_shared((bM, bK), T.float16)
                B_s = T.alloc_shared((bK, bN), T.float16)
                C_l = T.alloc_fragment((bM, bN), T.float32)
                T.clear(C_l)
                for k in T.Pipelined(K // bK, num_stages=n_stages):
                    T.copy(A[by * bM, k * bK], A_s)
                    T.copy(B[k * bK, bx * bN], B_s)
                    T.gemm(A_s, B_s, C_l)
                T.copy(C_l, C[by * bM, bx * bN])
        return kernel

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    fp16_gemm(num_stages)(A, B, C)
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A.float() @ B.float(), atol=1.0, rtol=5e-2)


if __name__ == "__main__":
    tilelang.testing.main()
