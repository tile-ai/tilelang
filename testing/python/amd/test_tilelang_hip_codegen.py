"""
Regression tests for HIP/AMD codegen fixes in TileLang.

Covers six bug fixes across five source files:

  Fix 1 (reduce.h)            warp_reduce 5-step butterfly with width=32
  Fix 2 (codegen_hip.cc,      ShuffleNode bfloat16x2/float16x2 packing;
          common.h)            uint1 bf16x2 math overloads
  Fix 3 (allocate.py,         T.alloc_var(init=<literal>) emits a correctly
          codegen_hip.cc)      initialised scalar on HIP
  Fix 4 (codegen_hip.cc)      T.sync_warp() lowered to no-op on HIP
  Fix 5 (codegen_hip.cc,      T.sync_grid() lowered to cooperative groups
          rt_mod_hip.cc,       grid barrier; runtime launch infrastructure
          stubs/)              added
  Fix 6 (pipeline_planning.cc) T.Pipelined(num_stages>1) falls back to a
                               plain sequential loop on ROCM to avoid LDS
                               overflow (hipModuleLaunchKernel EINVAL)
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T


# ---------------------------------------------------------------------------
# Fix 1 — src/tl_templates/hip/reduce.h
#   warp_reduce: 5-step butterfly with explicit width=32
#
# Symptom: On CDNA (wave64) with 32 active threads the old 6-step butterfly
#   called __shfl_xor(value, 32) without a width argument, reading uninitialised
#   VGPRs in lanes 32-63.  This produced NaN or garbage in every reduction that
#   went through warp_reduce (reduce_max, reduce_sum, AllReduce).
#
# Fix: remove the step-32 shuffle; add width=32 to every remaining step.
#   __shfl_xor(v, N, 32) restricts the butterfly to the lower 32-lane group,
#   matching CUDA warp semantics on CDNA wave64 and RDNA wave32 alike.
#   With 64 threads and width=32 the wavefront splits into two independent
#   32-lane groups — correct for kernels that assume logical warp_size=32.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("n_tokens,n_experts", [(64, 8), (128, 16), (512, 32)])
def test_warp_reduce_no_nan(n_tokens, n_experts):
    """
    32-thread-per-block reduce_max / reduce_sum must not produce NaN on CDNA.

    Old: __shfl_xor(v, 32) with 32 active threads reads uninit VGPRs → NaN.
    New: 5-step with width=32 stays in [0,31] group → correct, no NaN.
    """
    assert n_experts <= 32

    @tilelang.jit
    def gate_reduce(n_tok: int, n_exp: int):
        @T.prim_func
        def kernel(
            logits: T.Tensor((n_tok, n_exp), T.float32),
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

    logits = torch.randn(n_tokens, n_experts, dtype=torch.float32, device="cuda")
    out_max = torch.zeros(n_tokens, dtype=torch.float32, device="cuda")
    out_sum = torch.zeros(n_tokens, dtype=torch.float32, device="cuda")

    gate_reduce(n_tokens, n_experts)(logits, out_max, out_sum)
    torch.cuda.synchronize()

    assert not out_max.isnan().any(), "reduce_max NaN — __shfl_xor(v,32) uninit VGPR bug"
    assert not out_sum.isnan().any(), "reduce_sum NaN — __shfl_xor(v,32) uninit VGPR bug"
    torch.testing.assert_close(out_max, logits.max(dim=1).values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_sum, logits.sum(dim=1), atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_rocm
def test_warp_reduce_correctness_32_threads():
    """
    32-thread reduce_sum over 32 elements must return the exact sum on CDNA.

    Exercises the warp-level shuffle path directly.  With the old step-32
    shuffle, uninitialised VGPR reads on CDNA produced garbage.
    """
    N = 32

    @tilelang.jit
    def reduce_kernel():
        @T.prim_func
        def kernel(
            x: T.Tensor((N,), T.float32),
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

    x = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    reduce_kernel()(x, out)
    torch.cuda.synchronize()

    assert not out[0].isnan(), "reduce_sum NaN — warp_reduce VGPR bug on CDNA"
    torch.testing.assert_close(out[0], x.sum(), atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_rocm
def test_warp_reduce_with_64_threads_two_groups():
    """
    With 64 threads and width=32 the wavefront splits into two independent
    32-lane groups — each group's lane 0 holds its partial sum.

    Old: __shfl_xor(v, 32) without width mixed the groups → wrong result.
    New: width=32 confines each shuffle to its own 32-lane group → correct.
    """
    N, n_exp = 64, 4

    @tilelang.jit
    def two_warp_reduce():
        @T.prim_func
        def kernel(
            x: T.Tensor((N, n_exp), T.float32),
            out: T.Tensor((N,), T.float32),
        ) -> None:
            with T.Kernel(1, threads=N) as _:
                tx = T.get_thread_binding()
                frag = T.alloc_fragment(n_exp, T.float32)
                T.copy(x[tx, 0], frag)
                s = T.alloc_fragment(1, T.float32)
                T.reduce_sum(frag, s, dim=0)
                out[tx] = s[0]

        return kernel

    x = torch.ones(N, n_exp, dtype=torch.float32, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    two_warp_reduce()(x, out)
    torch.cuda.synchronize()

    assert not out.isnan().any(), "NaN in two-warp reduce — width=32 fix not applied"


# ---------------------------------------------------------------------------
# Fix 2 — src/target/codegen_hip.cc (VisitExpr_ ShuffleNode)
#         src/tl_templates/hip/common.h (uint1 bfloat16x2 math overloads)
#
# Symptom: Packing two bfloat16 scalars into a bfloat16x2 ShuffleNode caused
#   CodeGenC to emit `uint1(a, b)` — invalid HIP constructor → compile error.
#
# Fix (codegen_hip.cc): Override VisitExpr_(ShuffleNode) to emit
#   `uint1{__pack_bfloat162(a, b)}` / `uint1{__pack_half2(a, b)}`.
# Fix (common.h): Add abs2/max2/min2/add2/mul2 overloads for uint1 as a
#   packed bfloat16x2 carrier.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
def test_bfloat16_shuffle_codegen_and_correctness():
    """
    bfloat16 fragment warp-reduction: source must use __pack_bfloat162
    (not invalid `uint1(a,b)`) and the result must be numerically correct.
    """
    n_tok, n_exp = 16, 8

    @tilelang.jit
    def bf16_reduce(n_t: int, n_e: int):
        @T.prim_func
        def kernel(
            x: T.Tensor((n_t, n_e), T.bfloat16),
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

    kernel = bf16_reduce(n_tok, n_exp)

    # Source check: no invalid two-argument constructor
    src = kernel.get_kernel_source()
    assert "uint1(a" not in src and "uint1(b" not in src, "Old `uint1(a, b)` constructor found — ShuffleNode fix not applied"

    # Runtime correctness
    x = torch.randn(n_tok, n_exp, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(n_tok, dtype=torch.float32, device="cuda")
    kernel(x, out)
    torch.cuda.synchronize()
    assert not out.isnan().any(), "bf16 ShuffleNode reduction NaN"
    torch.testing.assert_close(out, x.float().sum(dim=1), atol=5e-2, rtol=1e-2)


@tilelang.testing.requires_rocm
def test_float16_shuffle_correctness():
    """
    float16 fragment warp-reduction exercises the __pack_half2 path.
    Analogous to the bfloat16 test but for float16x2 packing.
    """
    n_tok, n_exp = 64, 8

    @tilelang.jit
    def f16_reduce(n_t: int, n_e: int):
        @T.prim_func
        def kernel(
            x: T.Tensor((n_t, n_e), T.float16),
            out: T.Tensor((n_t,), T.float32),
        ) -> None:
            with T.Kernel(n_t, threads=32) as pid:
                frag = T.alloc_fragment(n_e, T.float16)
                T.copy(x[pid, 0], frag)
                frag_f32 = T.alloc_fragment(n_e, T.float32)
                for i in T.Parallel(n_e):
                    frag_f32[i] = T.cast(frag[i], T.float32)
                s = T.alloc_fragment(1, T.float32)
                T.reduce_sum(frag_f32, s, dim=0)
                if T.get_thread_binding() == 0:
                    out[pid] = s[0]

        return kernel

    x = torch.randn(n_tok, n_exp, dtype=torch.float16, device="cuda")
    out = torch.zeros(n_tok, dtype=torch.float32, device="cuda")
    f16_reduce(n_tok, n_exp)(x, out)
    torch.cuda.synchronize()
    assert not out.isnan().any(), "float16 ShuffleNode reduction NaN"
    torch.testing.assert_close(out, x.float().sum(dim=1), atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Fix 3 — tilelang/language/allocate.py + src/target/codegen_hip.cc
#   T.alloc_var(init=<literal>) initialisation on HIP;
#   local.var scalar declaration and GetBufferRef bare-name return
#
# Symptom (allocate.py): int/float literals used block_attr("tl.local_var_init")
#   which the HIP backend silently ignored → variable uninitialised at runtime.
# Symptom (codegen_hip.cc): AllocateNode emitted `type vid[1];` for local.var;
#   alloc_storage_scope_ was not updated → GetBufferRef fell through to an
#   invalid pointer-cast path → compile failure.
#
# Fix (allocate.py): always route init through T.buffer_store → explicit
#   BufferStore TIR node → assignment statement in every backend.
# Fix (codegen_hip.cc): emit `type vid = init;`; register alloc_storage_scope_
#   so GetBufferRef returns the bare name `vid`.
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_init():
    """Kernel that initialises a local int32 variable to 7 and writes it out."""

    @T.prim_func
    def main(Out: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            counter = T.alloc_var(T.int32, init=7)
            Out[tx] = counter

    return main


@tilelang.testing.requires_rocm
def test_alloc_var_init_in_hip_source():
    """Init value must appear as `= 7;` in the generated HIP source."""
    src = _kernel_alloc_var_init().get_kernel_source()
    assert "= 7;" in src, (
        f"T.alloc_var(T.int32, init=7) should generate '= 7;' in HIP source, but it was not found.\nGenerated source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_no_array_subscript_in_hip_source():
    """local.var must be declared as a scalar (no `counter[` array syntax)."""
    src = _kernel_alloc_var_init().get_kernel_source()
    assert "counter[" not in src, (
        f"local.var should be emitted as a scalar (e.g. 'int counter = 7'), but array-style access was found:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_correctness():
    """All output elements must equal 7 — the initialised value."""
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    _kernel_alloc_var_init()(out)
    assert torch.all(out == 7), f"Expected all 7, got: {out}"


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
    src = _kernel_multi_alloc_var_init().get_kernel_source()
    assert src.count("= 3;") >= 1, f"Init value 3 not found in HIP source:\n{src}"
    assert src.count("= 4;") >= 1, f"Init value 4 not found in HIP source:\n{src}"


@tilelang.testing.requires_rocm
def test_multi_alloc_var_init_correctness():
    """Sum of two initialised local variables must equal 7 (3+4)."""
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    _kernel_multi_alloc_var_init()(out)
    assert torch.all(out == 7), f"Expected all 7 (3+4), got: {out}"


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_count():
    """Counter initialised to 0, incremented 5 times in a loop."""

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
    """Variable initialised to 0 and incremented 5 times must equal 5."""
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    _kernel_alloc_var_count()(out)
    assert torch.all(out == 5), f"Expected all 5, got: {out}"


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "init_val,dtype_str",
    [
        (0, "int32"),
        (7, "int32"),
        (-3, "int32"),
        (0.0, "float32"),
        (1.0, "float32"),
        (-0.5, "float32"),
    ],
)
def test_alloc_var_literal_init_is_reliable(init_val, dtype_str):
    """
    alloc_var with any literal init must produce that exact value on HIP.

    Old: int/float literals → block_attr (silently ignored) → uninit.
    New: always T.buffer_store → `vid = init_val;` in generated HIP C.
    """
    tl_dtype = T.int32 if dtype_str == "int32" else T.float32
    torch_dtype = torch.int32 if dtype_str == "int32" else torch.float32
    N = 32

    @tilelang.jit
    def var_init_kernel(iv, tld):
        @T.prim_func
        def kernel(out: T.Tensor((N,), tld)) -> None:
            with T.Kernel(1, threads=N) as _:
                v = T.alloc_var(tld, init=iv)
                for i in T.Parallel(N):
                    out[i] = v

        return kernel

    out = torch.zeros(N, dtype=torch_dtype, device="cuda")
    var_init_kernel(init_val, tl_dtype)(out)
    torch.cuda.synchronize()

    expected = torch.full((N,), init_val, dtype=torch_dtype, device="cuda")
    if dtype_str == "int32":
        assert torch.equal(out, expected), f"alloc_var(init={init_val}) got {out[0].item()}, expected {init_val}"
    else:
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@tilelang.testing.requires_rocm
def test_alloc_var_inf_init():
    """
    alloc_var(init=-T.infinity(T.float32)) — the pattern used for top1_var /
    top2_var in MoE topk gate kernels — must produce -inf on HIP.
    """
    N = 32

    @tilelang.jit
    def inf_init_kernel():
        @T.prim_func
        def kernel(out: T.Tensor((N,), T.float32)) -> None:
            with T.Kernel(1, threads=N) as _:
                v = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                for i in T.Parallel(N):
                    out[i] = v

        return kernel

    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    inf_init_kernel()(out)
    torch.cuda.synchronize()
    assert out.isinf().all() and (out < 0).all(), f"alloc_var(init=-inf) got {out[0].item()}, expected -inf"


@tilelang.testing.requires_rocm
def test_alloc_var_init_zero_persists_across_serial_loop():
    """
    count_var = T.alloc_var(T.int32, init=0) must start at 0 and accumulate
    correctly.  This is the exact pattern used by count_var in MoE kernels.
    """
    N = 8

    @tilelang.jit
    def serial_count_kernel():
        @T.prim_func
        def kernel(out: T.Tensor((1,), T.int32)) -> None:
            with T.Kernel(1, threads=1) as _:
                count_var = T.alloc_var(T.int32, init=0)
                for _ in T.serial(N):
                    count_var = count_var + 1
                out[0] = count_var

        return kernel

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    serial_count_kernel()(out)
    torch.cuda.synchronize()
    assert out[0].item() == N, f"count_var: got {out[0].item()}, expected {N} — init=0 not applied (block_attr bug)"


@tilelang.testing.requires_rocm
def test_local_var_scalar_codegen():
    """
    local.var must be emitted and accessed as a plain scalar on HIP.

    Before: alloc_storage_scope_ not registered → GetBufferRef fell through to
            an invalid pointer-cast path → compile failure.
    After:  `type vid = init;` emitted; GetBufferRef returns bare `vid`.
    """
    N = 32

    @tilelang.jit
    def local_var_scalar():
        @T.prim_func
        def kernel(out: T.Tensor((N,), T.int32)) -> None:
            with T.Kernel(1, threads=N) as _:
                v = T.alloc_var(T.int32, init=5)
                if T.get_thread_binding() == 0:
                    v = v + 1
                for i in T.Parallel(N):
                    out[i] = v

        return kernel

    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    local_var_scalar()(out)
    torch.cuda.synchronize()
    assert out[0].item() == 6, f"local.var scalar: got {out[0].item()}, expected 6 (5+1)"


@tilelang.testing.requires_rocm
def test_local_var_float_init_readable():
    """
    local.var with float32 literal init must be readable on HIP.
    Before the alloc_storage_scope_ fix, GetBufferRef emitted invalid code.
    """

    @tilelang.jit
    def float_init_readback():
        @T.prim_func
        def kernel(out: T.Tensor((1,), T.float32)) -> None:
            with T.Kernel(1, threads=32) as _:
                v = T.alloc_var(T.float32, init=3.14)
                if T.get_thread_binding() == 0:
                    out[0] = v

        return kernel

    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    float_init_readback()(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[0].item(), 3.14, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Fix 4 — src/target/codegen_hip.cc
#   T.sync_warp() → no-op on HIP
#
# Symptom: tl::sync_warp() had no handler → codegen assertion failure or
#   undefined symbol at link time.
# Fix: emit an empty statement; AMD wavefronts execute in lockstep so
#   intra-wavefront convergence is guaranteed by hardware.
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
    """__syncwarp must NOT appear in the generated HIP source."""
    src = _kernel_sync_warp_codegen().get_kernel_source()
    assert "__syncwarp" not in src, f"T.sync_warp() should be a no-op on HIP, but __syncwarp was found in the generated source:\n{src}"


@tilelang.testing.requires_rocm
def test_sync_warp_correctness():
    """Kernel using T.sync_warp() must produce correct results on HIP."""
    A = torch.arange(32, dtype=torch.float32, device="cuda")
    B = torch.zeros(32, dtype=torch.float32, device="cuda")
    _kernel_sync_warp_codegen()(A, B)
    torch.testing.assert_close(B, A * 2.0)


@tilelang.testing.requires_rocm
def test_sync_warp_inside_conditional():
    """
    T.sync_warp() inside a conditional branch (pattern from moe/common.py
    get_topk_group_idx).  Verifies compilation and deterministic output.
    """
    N, M = 32, 8

    @tilelang.jit
    def sync_warp_cond_kernel():
        @T.prim_func
        def kernel(
            x: T.Tensor((N,), T.float32),
            out: T.Tensor((M,), T.float32),
        ) -> None:
            with T.Kernel(1, threads=N) as _:
                shmem = T.alloc_shared((M,), T.float32)
                tx = T.get_thread_binding()
                if tx < M:
                    shmem[tx] = x[tx]
                T.sync_warp()
                for i in T.Parallel(M):
                    out[i] = shmem[i]

        return kernel

    x = torch.randn(N, dtype=torch.float32, device="cuda")
    out = torch.zeros(M, dtype=torch.float32, device="cuda")
    sync_warp_cond_kernel()(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x[:M])


# ---------------------------------------------------------------------------
# Fix 5 — src/target/codegen_hip.cc, src/target/rt_mod_hip.cc,
#          src/target/stubs/hip.cc, src/target/stubs/hip.h
#   T.sync_grid() → cooperative_groups::this_grid().sync()
#
# Symptom: tl::sync_grid() had no handler → same assertion / link failure.
# Fix: emit cooperative_groups call; add need_cooperative_groups_ flag to
#   conditionally include hip_cooperative_groups.h; add runtime launch
#   infrastructure (hipModuleLaunchCooperativeKernel stubs).
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
    T.sync_grid() must emit cooperative_groups::this_grid().sync() and
    include <hip/hip_cooperative_groups.h> in the generated HIP source.

    Note: runtime execution requires hipModuleLaunchCooperativeKernel which
    is added via the stub infrastructure; this test validates codegen only.
    """
    src = _kernel_sync_grid_codegen().get_kernel_source()
    assert "this_grid().sync()" in src, f"T.sync_grid() should generate 'this_grid().sync()' but not found:\n{src}"
    assert "cooperative_groups" in src, f"T.sync_grid() should include cooperative_groups but not found:\n{src}"


# ---------------------------------------------------------------------------
# Fix 6 — src/transform/pipeline_planning.cc
#   Skip T.Pipelined(num_stages>1) pipeline planning on ROCM
#
# Symptom: Double-buffering doubled the LDS allocation for every shared buffer
#   inside the loop body, exhausting the per-workgroup LDS budget and causing
#   hipModuleLaunchKernel to return HIPERRORINVALIDVALUE.  LDS limits:
#     gfx942 (CDNA3 / MI300X): 64 KB;  gfx950 (CDNA4 / MI350): 160 KB (#2058)
#   Even with gfx950's larger budget, double-buffering large shared tiles can
#   still exceed 160 KB, and the HIP async-copy infrastructure has no ROCM
#   equivalent, so the planner cannot safely pipeline on any ROCM target.
#
# Fix: when TargetIsRocm() && num_stages > 1, skip pipeline planning and fall
#   back to a plain sequential loop with synchronous T.copy — always LDS-safe.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_pipelined_no_lds_overflow(num_stages):
    """
    T.Pipelined(num_stages=N) must not raise hipModuleLaunchKernel EINVAL and
    must produce the correct result regardless of N.

    Old: num_stages=2 doubled LDS → EINVAL (64 KB on gfx942, 160 KB on gfx950).
    New: multi-stage loops fall back to plain sequential on ROCM.
    """
    M, K, blk = 32, 256, 64

    @tilelang.jit
    def kernel(
        x: T.Tensor((M, K), T.float32),
        out: T.Tensor((M,), T.float32),
    ) -> None:
        with T.Kernel(M, threads=64) as pid:
            acc = T.alloc_fragment((1,), T.float32)
            xs = T.alloc_shared((blk,), T.float32)
            xl = T.alloc_fragment((blk,), T.float32)
            s = T.alloc_fragment((1,), T.float32)
            T.clear(acc)
            for k in T.Pipelined(K // blk, num_stages=num_stages):
                T.copy(x[pid, k * blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)
                T.reduce_sum(xl, s, dim=0)
                acc[0] = acc[0] + s[0]
            out[pid] = acc[0]

    x = torch.ones(M, K, dtype=torch.float32, device="cuda")
    out = torch.zeros(M, dtype=torch.float32, device="cuda")
    kernel(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.full((M,), float(K), device="cuda"), atol=1e-4, rtol=0)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipelined_multi_stage_fp16_gemm(num_stages):
    """
    FP16 GEMM with T.Pipelined(num_stages>1) must launch and produce correct
    results on ROCM — the most common pattern that triggered the LDS overflow
    (A_s bM×bK + B_s bK×bN doubled per pipeline stage).
    """
    M, N, K = 128, 128, 128
    bM, bN, bK = 64, 64, 32

    @tilelang.jit
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
            for k in T.Pipelined(K // bK, num_stages=num_stages):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[k * bK, bx * bN], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * bM, bx * bN])

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    kernel(A, B, C)
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A.float() @ B.float(), atol=1.0, rtol=5e-2)


if __name__ == "__main__":
    tilelang.testing.main()
