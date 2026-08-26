"""Reducer v2 (first-class deferred reduction epoch) Stage 1 tests.

Covers the FullParticipant wide baseline:
  - the Issue #2408 regression (36, not 576);
  - deferred accumulation across serial tiles;
  - multiple update sites with different parallel extents;
  - sum with DISTINCT contribution values (max/min are idempotent and
    cannot detect contribution-multiplicity bugs);
  - max/min/seed semantics;
  - epoch lifecycle / illegal access diagnostics.
"""

import re

import pytest
import tilelang
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import torch
import tvm

tilelang.testing.set_random_seed()


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_scalar_sum_regression_36_not_576():
    """Issue #2408: 128 threads summing 8 elements must yield 36, not 576.

    The T.Parallel(8) loop layout replicates every logical iteration over 16
    threads; layout replication must not change contribution multiplicity.
    """
    extent = 8
    threads = 128

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum().reshape(1), atol=0, rtol=0)


def test_deferred_sum_across_serial_tiles():
    """The reducer's core value: accumulate across serial tiles, one final
    cross-thread combine."""
    M, K, BLOCK_K, threads = 32, 512, 128, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            a_frag = T.alloc_fragment((M, BLOCK_K), T.float32)
            for k_tile in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[0, k_tile * BLOCK_K], a_frag)
                for i, k in T.Parallel(M, BLOCK_K):
                    T.reducer_update(acc[i], a_frag[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-2, rtol=1e-2)


def test_pipelined_accumulation():
    """reducer_update inside a T.Pipelined body: pipeline planning must
    classify the per-iteration intrinsic as compute (the shared-staging copy
    is the only copy stage), and the epoch spans all pipeline iterations
    with a single finalize.

    Warp specialization is disabled: a WS-split epoch (init/update/finalize
    inside one warp-group branch) is out of scope for reducer v2 and is
    rejected by a compile-time check in the finalize lowering."""
    M, K, BLOCK_K, threads = 32, 512, 128, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            a_shared = T.alloc_shared((M, BLOCK_K), T.float32)
            for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                T.copy(A[0, k_tile * BLOCK_K], a_shared)
                for i, k in T.Parallel(M, BLOCK_K):
                    T.reducer_update(acc[i], a_shared[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(
        kernel,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-2, rtol=1e-2)


def test_multiple_update_sites_mixed_extents():
    extent = 8
    threads = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((extent,), T.float32),
        B: T.Tensor((2 * extent,), T.float32),
        C: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((extent,), T.float32)
            b_frag = T.alloc_fragment((2 * extent,), T.float32)
            T.copy(A, a_frag)
            T.copy(B, b_frag)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], a_frag[i])
            for j in T.Parallel(2 * extent):
                T.reducer_update(acc[0], b_frag[j])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                C[0] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = torch.arange(1, 2 * extent + 1, dtype=torch.float32, device="cuda")
    C = tl.compile(kernel, out_idx=-1)(A, B)
    torch.testing.assert_close(C, (A.sum() + B.sum()).reshape(1), atol=0, rtol=0)


@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min(op):
    M, N, threads = 16, 64, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((M, N), T.float32)
            T.copy(A, a_frag)
            acc = T.alloc_reducer((M,), T.float32, op=op)
            T.reducer_init(acc)
            for i, j in T.Parallel(M, N):
                T.reducer_update(acc[i], a_frag[i, j])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    ref = A.max(dim=1).values if op == "max" else A.min(dim=1).values
    torch.testing.assert_close(B, ref, atol=0, rtol=0)


def test_seed_applied_exactly_once():
    """A sum seed must be combined once per logical output, not once per
    participant partial (which would multiply it by the thread count)."""
    extent, threads, seed = 8, 128, 100.0

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc, seed)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, (A.sum() + seed).reshape(1), atol=0, rtol=0)


def test_int32_sum():
    extent, threads = 8, 128

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.int32), B: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.int32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.int32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.int32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.int32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum().to(torch.int32).reshape(1), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Structural: physical plan codegen
# ---------------------------------------------------------------------------

_FORCE_BASELINE = {tilelang.PassConfigKey.TL_REDUCER_FORCE_BASELINE: True}


def _scalar_sum_kernel(extent=8, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    return kernel


def test_forced_baseline_codegen_uses_full_width_allreduce():
    threads = 128
    source = tl.compile(_scalar_sum_kernel(8, threads), out_idx=-1, pass_configs=_FORCE_BASELINE).get_kernel_source()
    assert "AllReduce" in source, source
    assert f"SumOp, {threads}" in source, source


def test_narrow_plan_scalar_sum_uses_projected_allreduce():
    """Parallel(8) -> acc[0] with 128 threads: the narrow plan reduces only
    over the 8 lanes that hold distinct contributions; the 16 replication
    groups each compute the full sum independently."""
    extent, threads = 8, 128
    source = tl.compile(_scalar_sum_kernel(extent, threads), out_idx=-1).get_kernel_source()
    assert f"SumOp, {extent}" in source, source
    assert f"SumOp, {threads}" not in source, source

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(_scalar_sum_kernel(extent, threads), out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum().reshape(1), atol=0, rtol=0)


def test_narrow_plan_local_complete_no_collective():
    """Parallel(M) -> acc[M]: each destination replica independently builds
    its complete result; zero collectives, barriers, or workspace."""
    M, threads = 8, 128

    @T.prim_func
    def kernel(A: T.Tensor((M,), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(M):
                T.reducer_update(acc[i], src[i])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "AllReduce" not in source, source

    A = torch.randn(M, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A, atol=0, rtol=0)


def test_narrow_plan_local_complete_inner_serial():
    """Parallel(M){ serial(K) }: the serial reduction accumulates on the
    owning thread; still zero collectives."""
    M, K, threads = 8, 16, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(M):
                for k in T.serial(K):
                    T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "AllReduce" not in source, source

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-3, rtol=1e-3)


def test_narrow_plan_row_reduction_projected_width():
    """Parallel(M, K) -> acc[M]: the collective combines only the K-sourced
    thread splits, not the full block."""
    M, K, threads = 16, 64, 256

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert f"SumOp, {threads}" not in source, source

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-3, rtol=1e-3)


def test_seed_with_narrow_plan():
    """Seeds work with narrow plans: combined exactly once per logical
    output after the projected collective, not once per replica group."""
    extent, threads, seed = 8, 128, 100.0

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc, seed)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert f"SumOp, {extent}" in source, source  # narrow plan fires
    assert f"SumOp, {threads}" not in source, source

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, (A.sum() + seed).reshape(1), atol=0, rtol=0)


def test_seed_with_local_complete():
    """Seed on a zero-collective (LocalComplete) epoch: the finalize still
    materializes to apply the seed once per slot."""
    M, threads, seed = 8, 128, 5.0

    @T.prim_func
    def kernel(A: T.Tensor((M,), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc, seed)
            for i in T.Parallel(M):
                T.reducer_update(acc[i], src[i])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "AllReduce" not in source, source

    A = torch.randn(M, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A + seed, atol=0, rtol=0)


def test_narrow_plan_permuted_indices():
    """Update indices may permute the parallel loop order: acc[j, i] with
    Parallel(i, j). The induced storage layout is rebuilt over the reducer's
    own dim order."""
    M, N, threads = 8, 16, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32), B: T.Tensor((N, M), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, N), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((N, M), T.float32, op="sum")
            T.reducer_init(acc)
            for i, j in T.Parallel(M, N):
                T.reducer_update(acc[j, i], src[i, j])
            result = T.alloc_fragment((N, M), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "AllReduce" not in source, source

    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.t().contiguous(), atol=0, rtol=0)


def test_narrow_plan_unit_dim_indices():
    """A constant zero index on a unit reducer dim mixes with loop-var
    indices: acc[0, i] with shape (1, M)."""
    M, K, threads = 16, 16, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((1, M), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1, M), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[0, i], src[i, k])
            result = T.alloc_fragment((1, M), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "AllReduce" in source, source
    assert f"SumOp, {threads}" not in source, source

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.sum(dim=1).reshape(1, M), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("force_baseline", [False, True])
def test_differential_forced_baseline_vs_auto(force_baseline):
    """Forced baseline and auto plan selection must agree numerically."""
    M, K, threads = 32, 128, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            a_frag = T.alloc_fragment((M, K), T.float32)
            T.copy(A, a_frag)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], a_frag[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    configs = _FORCE_BASELINE if force_baseline else None
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1, pass_configs=configs)(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Packed partial accumulation (16-bit floats)
# ---------------------------------------------------------------------------


def _rowsum_kernel(M, K, threads, dtype):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), dtype)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), dtype, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), dtype)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


def test_packed_accumulation_fp16_parallel_reduction():
    """fp16 narrow plans split each per-thread partial into two lanes keyed
    by the reduction var's parity ("_pk" storage), halving the serial combine
    dependence chain; a per-thread fold recombines the lanes before the
    (unchanged) projected collective.

    256-bit global vectorization (sm100+) is pinned off: it legitimately
    doubles the per-thread contiguous run to 16 halves and halves the
    projected collective to `SumOp, 4`, while the exact-width assertions
    below encode the 128-bit shape."""
    M, K, threads = 32, 64, 128
    configs = {tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True}
    source = tl.compile(_rowsum_kernel(M, K, threads, T.float16), out_idx=-1, pass_configs=configs).get_kernel_source()
    assert "_pk" in source, source
    assert "SumOp, 8" in source, source  # collective width unchanged by packing

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = tl.compile(_rowsum_kernel(M, K, threads, T.float16), out_idx=-1, pass_configs=configs)(A)
    torch.testing.assert_close(B, A.float().sum(dim=1).to(torch.float16), atol=1e-2, rtol=1e-2)


def test_packed_accumulation_fp16_inner_serial():
    """An enclosing serial loop is the preferred lane source: its parity
    alternates lanes on the owning thread with zero cross-thread effects."""
    M, K, threads = 8, 16, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((M,), T.float16)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float16)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float16, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(M):
                for k in T.serial(K):
                    T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.float16)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "_pk" in source, source
    assert "AllReduce" not in source, source  # still LocalComplete

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.float().sum(dim=1).to(torch.float16), atol=1e-2, rtol=1e-2)


def test_packed_accumulation_bf16_max():
    """Packing applies to idempotent combines on bf16 too (max folds are
    order-insensitive, so lanes cannot change the result)."""
    M, K, threads = 32, 64, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.bfloat16), B: T.Tensor((M,), T.bfloat16)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.bfloat16)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.bfloat16, op="max")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.bfloat16)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert "_pk" in source, source

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, A.max(dim=1).values, atol=0, rtol=0)


def test_fp32_narrow_plan_not_packed():
    """Packing is a 16-bit optimization: fp32 partials stay single-lane."""
    M, K, threads = 32, 64, 128
    source = tl.compile(_rowsum_kernel(M, K, threads, T.float32), out_idx=-1).get_kernel_source()
    assert "_pk" not in source, source


@pytest.mark.parametrize("force_baseline", [False, True])
def test_differential_packed_vs_baseline_fp16(force_baseline):
    """The packed narrow plan and the wide baseline must agree numerically
    (fp16 sum: same contribution multiset up to reassociation)."""
    M, K, threads = 32, 64, 128
    configs = _FORCE_BASELINE if force_baseline else None
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = tl.compile(_rowsum_kernel(M, K, threads, T.float16), out_idx=-1, pass_configs=configs)(A)
    torch.testing.assert_close(B, A.float().sum(dim=1).to(torch.float16), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Bitwise combine ops
# ---------------------------------------------------------------------------


def _bitwise_rowreduce_kernel(M, K, threads, op):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.int32), B: T.Tensor((M,), T.int32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.int32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.int32, op=op)
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.int32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


def _bitwise_ref(A, op):
    out = None
    for col in range(A.shape[1]):
        cur = A[:, col]
        if out is None:
            out = cur.clone()
        elif op == "bitand":
            out &= cur
        elif op == "bitor":
            out |= cur
        elif op == "bitxor":
            out ^= cur
    return out


@pytest.mark.parametrize("op", ["bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("force_baseline", [False, True])
def test_bitwise_reduce(op, force_baseline):
    """Bitwise combines under both plans. The narrow plan's collective and
    the wide baseline must agree bit-for-bit (bitwise ops are associative,
    commutative and exact)."""
    M, K, threads = 16, 64, 128
    configs = _FORCE_BASELINE if force_baseline else None
    A = torch.randint(0, 2**31 - 1, (M, K), dtype=torch.int32, device="cuda")
    B = tl.compile(_bitwise_rowreduce_kernel(M, K, threads, op), out_idx=-1, pass_configs=configs)(A)
    torch.testing.assert_close(B, _bitwise_ref(A, op), atol=0, rtol=0)


def test_bitwise_reducer_rejects_float_dtype():
    with pytest.raises(AssertionError, match="integer dtype"):

        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                acc = T.alloc_reducer((1,), T.float32, op="bitand")
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], A[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                B[0] = result[0]


# ---------------------------------------------------------------------------
# Lifecycle / access diagnostics
# ---------------------------------------------------------------------------


def _compile_expect_error(kernel_factory, match):
    with pytest.raises(Exception, match=match):
        kernel = kernel_factory()
        tl.compile(kernel, out_idx=-1)


def test_reject_init_value_dtype_mismatch():
    """The T.reducer_init starting value must already have the reducer's
    dtype when passed as a PrimExpr (Python numbers are auto-converted)."""

    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc, T.float16(1.0))
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "does not match reducer")


def test_reject_update_before_init():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "before T.reducer_init")


def test_reject_missing_finalize():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                if T.get_thread_binding() == 0:
                    B[0] = 0.0

        return kernel

    _compile_expect_error(make, "never finalized")


def test_reject_ordinary_store():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    acc[0] += src[i]  # v1 syntax: must be rejected
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "illegal (store to|read of) reducer")


def test_reject_read_before_finalize():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                if T.get_thread_binding() == 0:
                    B[0] = acc[0]  # reading a partial: must be rejected
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)

        return kernel

    _compile_expect_error(make, "illegal read of reducer")


def test_reject_clear_on_reducer():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.clear(acc)  # v1 epoch open: must be rejected
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "(illegal|reducer)")


def test_reject_update_outside_parallel():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                T.reducer_update(acc[0], src[0])  # thread-uniform: ambiguous
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "outside any T.Parallel")


def test_reject_double_init():
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "double T.reducer_init")


# ---------------------------------------------------------------------------
# Epochs inside thread-uniform control flow (reopen once per iteration)
# ---------------------------------------------------------------------------


def test_epoch_inside_serial_loop_online_softmax():
    """A full epoch (init/update/finalize) inside a serial tile loop reopens
    once per iteration — the online-softmax rescale pattern. The max epoch
    seeds from the running maximum, a loop-variant expression."""
    K, BLOCK_K, threads = 512, 128, 128

    @T.prim_func
    def kernel(A: T.Tensor((K,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((BLOCK_K,), T.float32)
            running_max = T.alloc_fragment((1,), T.float32)
            running_sum = T.alloc_fragment((1,), T.float32)
            running_max[0] = -T.infinity(T.float32)
            running_sum[0] = 0.0
            for k_tile in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[k_tile * BLOCK_K], a_frag)
                new_max = T.alloc_reducer((1,), T.float32, op="max")
                T.reducer_init(new_max, running_max[0])
                for i in T.Parallel(BLOCK_K):
                    T.reducer_update(new_max[0], a_frag[i])
                max_result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(new_max, max_result)

                new_sum = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(new_sum)
                for i in T.Parallel(BLOCK_K):
                    T.reducer_update(new_sum[0], T.exp(a_frag[i] - max_result[0]))
                sum_result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(new_sum, sum_result)

                running_sum[0] = T.exp(running_max[0] - max_result[0]) * running_sum[0] + sum_result[0]
                running_max[0] = max_result[0]
            if T.get_thread_binding() == 0:
                B[0] = T.log(running_sum[0]) + running_max[0]

    A = torch.randn(K, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, torch.logsumexp(A, dim=0).reshape(1), atol=1e-4, rtol=1e-4)


def test_epoch_inside_pipelined_loop():
    """A full epoch inside a T.Pipelined body: the per-iteration collective
    must survive software pipelining. Warp specialization is disabled, as
    for every reducer-under-pipelining test."""
    K, BLOCK_K, threads = 512, 128, 128

    @T.prim_func
    def kernel(A: T.Tensor((K,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_shared = T.alloc_shared((BLOCK_K,), T.float32)
            a_frag = T.alloc_fragment((BLOCK_K,), T.float32)
            running_max = T.alloc_fragment((1,), T.float32)
            running_max[0] = -T.infinity(T.float32)
            for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                T.copy(A[k_tile * BLOCK_K], a_shared)
                T.copy(a_shared, a_frag)
                new_max = T.alloc_reducer((1,), T.float32, op="max")
                T.reducer_init(new_max, running_max[0])
                for i in T.Parallel(BLOCK_K):
                    T.reducer_update(new_max[0], a_frag[i])
                max_result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(new_max, max_result)
                running_max[0] = max_result[0]
            if T.get_thread_binding() == 0:
                B[0] = running_max[0]

    A = torch.randn(K, dtype=torch.float32, device="cuda")
    B = tl.compile(
        kernel,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )(A)
    torch.testing.assert_close(B, A.max().reshape(1), atol=0, rtol=0)


def test_epoch_inside_uniform_conditional():
    """A full epoch inside a block-uniform conditional: every thread of the
    block takes the same branch, so the collective stays barrier-uniform."""
    extent, threads = 8, 128

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((2,), T.float32)):
        with T.Kernel(2, threads=threads) as bx:
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            result = T.alloc_fragment((1,), T.float32)
            result[0] = 0.0
            if bx == 0:
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(extent):
                    T.reducer_update(acc[0], src[i])
                T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[bx] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    expected = torch.stack([A.sum(), torch.zeros((), device="cuda")])
    torch.testing.assert_close(B, expected, atol=0, rtol=0)


def test_seed_captured_at_init_site():
    """The T.reducer_init starting value is evaluated at the init site:
    overwriting the expression's source before finalize must not change the
    epoch's logical starting value."""
    extent, threads = 8, 128

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            seed = T.alloc_fragment((1,), T.float32)
            seed[0] = 5.0
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc, seed[0])
            seed[0] = 100.0  # after init, before finalize: must not matter
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A)
    torch.testing.assert_close(B, (A.sum() + 5.0).reshape(1), atol=0, rtol=0)


def test_reject_finalize_in_different_scope():
    """Init inside a serial loop with the finalize outside is rejected: the
    finalize is not in the init's scope (nor a conditional refinement)."""

    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                for _ in T.serial(4):
                    T.reducer_init(acc)
                    for i in T.Parallel(8):
                        T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "not in the scope of its T.reducer_init")


def test_finalize_in_conditional_refinement():
    """A finalize inside a block-uniform conditional nested in the init's
    scope runs 0 or 1 times per init — legal. Blocks that skip the branch
    leave the partials unread. (The sum_smaller_probs pattern.)"""
    extent, threads = 8, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((extent,), T.float32),
        S: T.Tensor((2,), T.int32),
        B: T.Tensor((2,), T.float32),
    ):
        with T.Kernel(2, threads=threads) as bx:
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            if S[bx] < 0:
                if T.get_thread_binding() == 0:
                    B[bx] = 0.0
            else:
                for i in T.Parallel(extent):
                    T.reducer_update(acc[0], src[i])
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[bx] = result[0]

    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    S = torch.tensor([-1, 1], dtype=torch.int32, device="cuda")
    B = tl.compile(kernel, out_idx=-1)(A, S)
    expected = torch.stack([torch.zeros((), device="cuda"), A.sum()])
    torch.testing.assert_close(B, expected, atol=0, rtol=0)


def test_reject_finalize_in_extra_loop():
    """A finalize inside a loop the init is not in would rerun the collective
    on already-combined partials — rejected even though the init context is
    a prefix of the finalize context."""

    def make():
        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                result = T.alloc_fragment((1,), T.float32)
                for _ in T.serial(4):
                    for i in T.Parallel(8):
                        T.reducer_update(acc[0], src[i])
                    T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]

        return kernel

    _compile_expect_error(make, "not in the scope of its T.reducer_init")


# ---------------------------------------------------------------------------
# Wide-plan finalize publication (thread-indexed readback regression)
# ---------------------------------------------------------------------------
#
# Serial loop vars driving the update indices reject the narrow plan, so the
# epoch takes the wide FullParticipant baseline. The finalize publishes
# `accumulator -> result` with a tl.copy; under the free-level distributed
# layout LayoutInference picked for `result` (serving only the later
# `result -> global` copy), that copy reads the fully replicated accumulator
# registers at thread-dependent physical indices and ptxas lowers the whole
# array to local memory. The planner must re-layout the unconstrained
# destination chain to participant-wide replication instead: static register
# indices everywhere and a replica-zero guard on the global store.


def _assert_static_wide_publication(source, local_arrays):
    for name in local_arrays:
        assert not re.search(rf"\b{name}\[[^\]\n]*threadIdx", source), source
    assert "if (((int)threadIdx.x) == 0)" in source, source


def _wide_plan_outer_product_kernel(K=256, threads=128):
    """acc[i, j] += src[i, k] * src2[j, k] with serial i, j: wide plan."""

    @T.prim_func
    def kernel(
        A: T.Tensor((4, K), T.float32),
        A2: T.Tensor((4, K), T.float32),
        B: T.Tensor((4, 4), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            src2 = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            T.copy(A2, src2)
            acc = T.alloc_reducer((4, 4), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.Parallel(K):
                        T.reducer_update(acc[i, j], src[i, k] * src2[j, k])
            result = T.alloc_fragment((4, 4), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


def test_wide_plan_finalize_publication_static_indices_2d():
    K = 256
    kernel = tl.compile(_wide_plan_outer_product_kernel(K), out_idx=-1)
    _assert_static_wide_publication(kernel.get_kernel_source(), ["acc", "result"])

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    A2 = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B = kernel(A, A2)
    torch.testing.assert_close(B, A @ A2.T, atol=1e-2, rtol=1e-2)


def test_wide_plan_finalize_publication_static_indices_1d():
    K, threads = 256, 128

    @T.prim_func
    def kernel(A: T.Tensor((4, K), T.float32), B: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((4,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.serial(4):
                for k in T.Parallel(K):
                    T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((4,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    compiled = tl.compile(kernel, out_idx=-1)
    _assert_static_wide_publication(compiled.get_kernel_source(), ["acc", "result"])

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B = compiled(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=1e-2, rtol=1e-2)


def test_wide_plan_finalize_publication_legacy_entry():
    """The legacy (v1) syntax reaches the same wide plan through
    CanonicalizeLegacyReducer's fresh `acc_result` destination."""
    K, threads = 256, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((4, K), T.float32),
        A2: T.Tensor((4, K), T.float32),
        B: T.Tensor((4, 4), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            src2 = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            T.copy(A2, src2)
            acc = T.alloc_reducer((4, 4), T.float32, replication="all")
            T.clear(acc)
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.Parallel(K):
                        acc[i, j] += src[i, k] * src2[j, k]
            T.finalize_reducer(acc)
            T.copy(acc, B)

    compiled = tl.compile(kernel, out_idx=-1)
    _assert_static_wide_publication(compiled.get_kernel_source(), ["acc", "acc_result"])

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    A2 = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B = compiled(A, A2)
    torch.testing.assert_close(B, A @ A2.T, atol=1e-2, rtol=1e-2)


def test_wide_plan_finalize_staged_chain_override():
    """`result` feeds a dtype-converting staging fragment before global: the
    whole unconstrained chain is re-layouted together, so both hops keep
    static register indices."""
    K, threads = 256, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((4, K), T.float32),
        A2: T.Tensor((4, K), T.float32),
        B: T.Tensor((4, 4), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            src2 = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            T.copy(A2, src2)
            acc = T.alloc_reducer((4, 4), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.Parallel(K):
                        T.reducer_update(acc[i, j], src[i, k] * src2[j, k])
            result = T.alloc_fragment((4, 4), T.float32)
            T.finalize_reducer(acc, result)
            staged = T.alloc_fragment((4, 4), T.float16)
            T.copy(result, staged)
            T.copy(staged, B)

    compiled = tl.compile(kernel, out_idx=-1)
    _assert_static_wide_publication(compiled.get_kernel_source(), ["acc", "result", "staged"])

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    A2 = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B = compiled(A, A2)
    torch.testing.assert_close(B, (A @ A2.T).to(torch.float16), atol=1e-1, rtol=1e-2)


def test_wide_steering_static_accumulator_with_compute_consumer():
    """A wide-plan destination with a consumer beyond the copy chain: the
    materializer's override proof must leave a consumer-chosen layout alone,
    so before dst-steering the publish copy gathered the replicated `acc`
    at thread-dependent indices and ptxas demoted the hot accumulator to
    local memory. finalize's kFree proposal now replicates the unconstrained
    dst up front: `acc` stays statically indexed (registers) and publishes
    through the replica-zero path; the residual dynamic read moves into the
    cold consumer loop (`result`, written once)."""
    K, threads = 256, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((4, K), T.float32),
        A2: T.Tensor((4, K), T.float32),
        B: T.Tensor((4, 4), T.float32),
        C: T.Tensor((4, 4), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            src2 = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            T.copy(A2, src2)
            acc = T.alloc_reducer((4, 4), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.Parallel(K):
                        T.reducer_update(acc[i, j], src[i, k] * src2[j, k])
            result = T.alloc_fragment((4, 4), T.float32)
            T.finalize_reducer(acc, result)
            doubled = T.alloc_fragment((4, 4), T.float32)
            for i, j in T.Parallel(4, 4):
                doubled[i, j] = result[i, j] * T.float32(2.0)
            T.copy(result, B)
            T.copy(doubled, C)

    compiled = tl.compile(kernel, out_idx=[-2, -1])
    source = compiled.get_kernel_source()
    assert not re.search(r"\bacc\[[^\]\n]*threadIdx", source), source
    assert "if (((int)threadIdx.x) == 0)" in source, source

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    A2 = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B, C = compiled(A, A2)
    ref = A @ A2.T
    torch.testing.assert_close(B, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(C, ref * 2, atol=1e-2, rtol=1e-2)


def test_narrow_steering_dst_compute_consumer():
    """dst-steering main case: `result` is consumed by a compute Parallel
    loop, not a copy chain. Free mode used to hand it that consumer's
    arbitrary layout, destination containment failed, and the epoch
    downgraded to the wide baseline (participant-wide AllReduce plus a
    shared workspace). finalize's kFree proposal now steers the
    unconstrained dst to the update sites' induced layout, so the narrow
    plan survives a compute consumer."""
    M, K, threads = 16, 64, 256

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), C: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            out = T.alloc_fragment((M,), T.float32)
            for i in T.Parallel(M):
                out[i] = result[i] * T.float32(2.0)
            T.copy(out, C)

    compiled = tl.compile(kernel, out_idx=-1)
    source = compiled.get_kernel_source()
    assert f"SumOp, {threads}" not in source, source  # not the wide baseline
    assert "SumOp, 16" in source, source  # collective over the K-splits only
    assert "workspace" not in source, source

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    C = compiled(A)
    torch.testing.assert_close(C, A.sum(dim=1) * 2, atol=1e-3, rtol=1e-3)


def test_annotated_dst_not_steered():
    """An annotated destination layout is authoritative: steering only fires
    for unconstrained fragments. The compact annotation stands (one element
    on each of 16 threads — a single physical slot), even though steering
    would have replicated it, and the extra compute consumer keeps the
    materializer's override proof from replacing it either."""
    K, threads = 256, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((4, K), T.float32),
        A2: T.Tensor((4, K), T.float32),
        B: T.Tensor((4, 4), T.float32),
        C: T.Tensor((4, 4), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((4, K), T.float32)
            src2 = T.alloc_fragment((4, K), T.float32)
            T.copy(A, src)
            T.copy(A2, src2)
            acc = T.alloc_reducer((4, 4), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.Parallel(K):
                        T.reducer_update(acc[i, j], src[i, k] * src2[j, k])
            result = T.alloc_fragment((4, 4), T.float32)
            T.annotate_layout({result: T.Fragment(result.shape, forward_fn=lambda a, b: (a * 4 + b, 0))})
            T.finalize_reducer(acc, result)
            doubled = T.alloc_fragment((4, 4), T.float32)
            for i, j in T.Parallel(4, 4):
                doubled[i, j] = result[i, j] * T.float32(2.0)
            T.copy(result, B)
            T.copy(doubled, C)

    compiled = tl.compile(kernel, out_idx=[-2, -1])
    source = compiled.get_kernel_source()
    assert "float result[1];" in source, source

    A = torch.randn(4, K, dtype=torch.float32, device="cuda")
    A2 = torch.randn(4, K, dtype=torch.float32, device="cuda")
    B, C = compiled(A, A2)
    ref = A @ A2.T
    torch.testing.assert_close(B, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(C, ref * 2, atol=1e-2, rtol=1e-2)


def test_narrow_steering_unsatisfiable_falls_back_wide():
    """A reserved dst takes its layout from finalize's verdict, but honoring
    can be genuinely unsatisfiable: here the consumer loop also reads a
    fragment annotated to a layout that conflicts with the induced (narrow)
    verdict, so every free-mode ordering dies on re-validation. The engine
    must then retry with the universally readable wide fallback (replicated
    dst) instead of failing with "no available layout found"."""
    M, K, threads = 16, 64, 256

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float32),
        O: T.Tensor((M,), T.float32),
        C: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, K), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], src[i, k])
            result = T.alloc_fragment((M,), T.float32)
            other = T.alloc_fragment((M,), T.float32)
            T.annotate_layout({other: T.Fragment(other.shape, forward_fn=lambda a: (a, 0))})
            T.copy(O, other)
            T.finalize_reducer(acc, result)
            out = T.alloc_fragment((M,), T.float32)
            for i in T.Parallel(M):
                out[i] = result[i] * T.float32(2.0) + other[i]
            T.copy(out, C)

    compiled = tl.compile(kernel, out_idx=-1)
    source = compiled.get_kernel_source()
    # Wide fallback signature: replicated dst, participant-wide collective.
    assert f"float result[{M}];" in source, source
    assert re.search(r"SumOp, 256", source), source

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    O = torch.randn(M, dtype=torch.float32, device="cuda")
    C = compiled(A, O)
    torch.testing.assert_close(C, A.sum(dim=1) * 2 + O, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Vectorized reducer updates: the combine store is an ordinary RMW to the
# vectorizer. Output-axis contiguous updates vectorize (and 16-bit/f32x2
# dtypes take the packed-math fast path in codegen); reduction-axis updates
# are kept scalar by the planner's invariant-store clamp. A float4/float2
# LOAD DECLARATION of `acc` discriminates a vectorized RMW from the identity
# fill and the finalize publish copy, which never declare vector loads of it.
# ---------------------------------------------------------------------------

_VECTOR_RMW_RE = r"float\d+ v\w* = \*\(float\d+\*\)\(acc"


def _contiguous_update_kernel(M, threads, dtype, run, op="sum"):
    layout = T.Fragment((M,), forward_fn=lambda i: (i // run, i % run))

    @T.prim_func
    def kernel(A: T.Tensor((M,), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=threads):
            a_buf = T.alloc_shared((M,), dtype)
            T.copy(A, a_buf)
            acc = T.alloc_reducer((M,), dtype, op=op)
            T.reducer_init(acc)
            for i in T.Parallel(M, loop_layout=layout):
                T.reducer_update(acc[i], a_buf[i])
            result = T.alloc_fragment((M,), dtype)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


def test_update_vectorizes_on_contiguous_layout():
    """Narrow plan, thread-contiguous output runs: the combine RMW lowers to
    a vector load / add / store of the accumulator."""
    M, threads = 512, 128
    kern = tl.compile(_contiguous_update_kernel(M, threads, T.float32, run=4), out_idx=-1)
    src = kern.get_kernel_source()
    assert re.search(_VECTOR_RMW_RE, src), src
    A = torch.randn(M, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A, atol=0, rtol=0)


def test_update_vectorizes_wide_plan():
    """The forced FullParticipant baseline vectorizes too: wide combine
    stores are plain RMWs once the multiplicity marker is lowered, and the
    former blanket vectorization exclusion is gone."""
    M, threads = 256, 64
    kern = tl.compile(
        _contiguous_update_kernel(M, threads, T.float32, run=4),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_REDUCER_FORCE_BASELINE: True},
    )
    src = kern.get_kernel_source()
    assert re.search(_VECTOR_RMW_RE, src), src
    A = torch.randn(M, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A, atol=0, rtol=0)


def test_update_vectorize_attempt_keeps_replica_guard():
    """A wide epoch whose loop layout replicates (8 iterations on 128
    threads) carries a replica guard; attempting vectorization on the loop
    must not change contribution multiplicity. 36, not 576."""
    extent, threads = 8, 128

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_buf = T.alloc_shared((extent,), T.float32)
            T.copy(A, a_buf)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], a_buf[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    kern = tl.compile(
        kernel,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_REDUCER_FORCE_BASELINE: True},
    )
    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A.sum().reshape(1), atol=0, rtol=0)


def test_update_reduction_axis_stays_scalar():
    """When the per-thread serial axis is the reduction axis (the combine
    store's index does not advance with it), the planner must keep the RMW
    scalar — vectorizing it would collapse the dependent chain."""
    M, K, threads = 4, 128, 128
    layout = T.Fragment((M, K), forward_fn=lambda i, k: (i * 32 + k // 4, k % 4))

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_buf = T.alloc_shared((M, K), T.float32)
            T.copy(A, a_buf)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K, loop_layout=layout):
                T.reducer_update(acc[i], a_buf[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    kern = tl.compile(kernel, out_idx=-1)
    src = kern.get_kernel_source()
    assert re.search(_VECTOR_RMW_RE, src) is None, src
    A = torch.arange(M * K, dtype=torch.float32, device="cuda").reshape(M, K)
    torch.testing.assert_close(kern(A), A.sum(dim=1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("op", "packed_fn"), [("sum", "add2"), ("max", "max2")])
def test_update_packed_fp16(op, packed_fn):
    """A vectorized fp16 combine takes the packed-math fast path: codegen
    emits tl::add2/max2 (__hadd2/__hmax2) pairs instead of per-lane scalar
    half ops. The vectorization itself is op-agnostic; the packed fast path
    covers add/sub/mul/fma/min/max."""
    M, threads = 512, 128
    kern = tl.compile(_contiguous_update_kernel(M, threads, T.float16, run=4, op=op), out_idx=-1)
    src = kern.get_kernel_source()
    assert packed_fn in src, src
    A = torch.randn(M, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(kern(A), A, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_update_packed_fp32x2_sm100_codegen():
    """On SM100+, a 2-lane fp32 combine takes the f32x2 packed-math path
    (fadd2). Codegen-only: lower for sm_100a and inspect the source."""
    target = {"kind": "cuda", "arch": "sm_100a"}
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(_contiguous_update_kernel(256, 128, T.float32, run=2), target=target)
    assert "add2" in artifact.kernel_source, artifact.kernel_source


def test_partial_layout_widen_on_disagreeing_sites():
    """PartialFragment widen-on-conflict: two individually narrow-eligible
    update nests whose induced partial layouts disagree must converge to the
    participant-wide plan (never a layout-conflict fatal), with exact sums."""
    M, threads = 32, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((M, 4), T.float32),
        B: T.Tensor((M, 8), T.float32),
        C: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((M, 4), T.float32)
            b_frag = T.alloc_fragment((M, 8), T.float32)
            T.copy(A, a_frag)
            T.copy(B, b_frag)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            for i, k in T.Parallel(M, 4):
                T.reducer_update(acc[i], a_frag[i, k])
            for i, k in T.Parallel(M, 8):
                T.reducer_update(acc[i], b_frag[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, C)

    A = torch.randn(M, 4, dtype=torch.float32, device="cuda")
    B = torch.randn(M, 8, dtype=torch.float32, device="cuda")
    C = tl.compile(kernel, out_idx=-1)(A, B)
    torch.testing.assert_close(C, A.sum(dim=1) + B.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_zero_update_epoch_wide_floor():
    """An epoch with no update site has no partial-layout proposer among the
    update nests; the engine seeds the participant-wide floor so finalize can
    still steer its destination, and the result is exactly the seed."""
    threads = 128

    @T.prim_func
    def kernel(B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc, T.float32(3.5))
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    B = tl.compile(kernel, out_idx=-1)()
    torch.testing.assert_close(B, torch.full((1,), 3.5, dtype=torch.float32, device="cuda"), atol=0, rtol=0)


def _annotated_partial_kernel(annotate: bool, lanes: int):
    """GEMV-shaped kernel; when `annotate` is set, pin the reducer to a
    middle plan: thread (i % (128 // lanes)) * lanes + rep, i.e. `lanes`
    combine lanes per row, 32 * lanes / 128 partials per thread."""
    M, K, threads = 32, 32, 128
    groups = threads // lanes

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((M, K), T.float32)
            T.copy(A, a_frag)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            if annotate:
                T.annotate_layout(
                    {
                        acc: T.PartialFragment(
                            (M,),
                            forward_thread_fn=lambda i, rep: (i % groups) * lanes + rep,
                            replicate=lanes,
                        )
                    }
                )
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], a_frag[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


# The automatic plan for _annotated_partial_kernel depends on the target's
# vector width (128-bit loads -> 8 lanes per row; sm100+ 256-bit loads -> 4
# lanes), so the exact-width assertions below pin 256-bit vectorization off
# and encode the 128-bit shape: auto = 8 lanes, annotation = 4.
_ANNOTATED_PARTIAL_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True}


def test_annotated_partial_middle_plan():
    """A user-annotated PartialFragment pins a middle plan the automatic
    solution would not pick (auto solves 8 lanes per row under the pinned
    128-bit vectorization): 4 combine lanes per row, so finalize emits an
    AllReduce of width exactly 4."""
    kern = tl.compile(_annotated_partial_kernel(annotate=True, lanes=4), out_idx=-1, pass_configs=_ANNOTATED_PARTIAL_CONFIGS)
    src = kern.get_kernel_source()
    assert "AllReduce<tl::SumOp, 4," in src, src
    A = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_annotated_partial_steers_inputs():
    """The annotated partial back-propagates through the update loop into
    the input fragment's layout: the baseline auto-solves 8 lanes per row
    (pinned 128-bit vectorization), the annotation forces 4, and both stay
    numerically exact."""
    base = tl.compile(_annotated_partial_kernel(annotate=False, lanes=4), out_idx=-1, pass_configs=_ANNOTATED_PARTIAL_CONFIGS)
    pinned = tl.compile(_annotated_partial_kernel(annotate=True, lanes=4), out_idx=-1, pass_configs=_ANNOTATED_PARTIAL_CONFIGS)
    assert "AllReduce<tl::SumOp, 8," in base.get_kernel_source(), base.get_kernel_source()
    assert "AllReduce<tl::SumOp, 4," in pinned.get_kernel_source(), pinned.get_kernel_source()
    A = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(base(A), A.sum(dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pinned(A), A.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_annotated_partial_conflict_raises():
    """An annotated input fragment forces a loop layout whose induced
    partial disagrees with the annotated reducer partial: the update site
    reports the conflict."""
    M, K, threads = 32, 32, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((M, K), T.float32)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.annotate_layout(
                {
                    # a_frag: each row spread over 8 lanes of 4 consecutive k.
                    a_frag: T.Fragment(
                        (M, K),
                        forward_thread_fn=lambda i, k: (i % 16) * 8 + k // 4,
                    ),
                    # acc: 4 lanes per row — incompatible with a_frag's 8 lanes.
                    acc: T.PartialFragment(
                        (M,),
                        forward_thread_fn=lambda i, rep: (i % 32) * 4 + rep,
                        replicate=4,
                    ),
                }
            )
            T.copy(A, a_frag)
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], a_frag[i, k])
            result = T.alloc_fragment((M,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    with pytest.raises(Exception, match="annotated"):
        tl.compile(kernel, out_idx=-1)


def test_annotate_plain_fragment_on_reducer_raises():
    """The frontend rejects a plain Fragment on a reducer buffer: its
    replicas are addends, not equal copies."""
    M, threads = 32, 128

    def build():

        @T.prim_func
        def kernel(A: T.Tensor((M,), T.float32), B: T.Tensor((M,), T.float32)):
            with T.Kernel(1, threads=threads):
                acc = T.alloc_reducer((M,), T.float32, op="sum")
                T.annotate_layout({acc: T.Fragment((M,), forward_thread_fn=lambda i: i)})
                T.reducer_init(acc)
                for i in T.Parallel(M):
                    T.reducer_update(acc[i], T.float32(1.0))
                result = T.alloc_fragment((M,), T.float32)
                T.finalize_reducer(acc, result)
                T.copy(result, B)

        return kernel

    with pytest.raises(Exception, match="PartialFragment"):
        build()


def test_annotated_dst_incompatible_unconstrained_chain_overridden():
    """An annotated finalize destination whose placement is incompatible
    with the reduction's natural placement, but whose whole use chain is
    unconstrained (finalize-written, read only by the copy to global), is
    re-layouted to the induced placement: the narrow plan survives and the
    numerics stay exact."""
    M, K, threads = 32, 32, 128

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((M, K), T.float32)
            T.copy(A, a_frag)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            result = T.alloc_fragment((M,), T.float32)
            # Each dst element owned by exactly one thread — thread i does
            # not hold element i's combined value under the narrow plan.
            T.annotate_layout(
                {
                    result: T.Fragment((M,), forward_thread_fn=lambda i: i),
                }
            )
            T.reducer_init(acc)
            for i, k in T.Parallel(M, K):
                T.reducer_update(acc[i], a_frag[i, k])
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    # 128-bit vectorization pinned: the narrow collective width is 8 on
    # every arch (256-bit loads would halve it to 4 on sm100+).
    kern = tl.compile(kernel, out_idx=-1, pass_configs=_ANNOTATED_PARTIAL_CONFIGS)
    src = kern.get_kernel_source()
    assert "SumOp, 8," in src, src  # narrow plan survives the override
    assert f"SumOp, {threads}" not in src, src
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A.sum(dim=1), atol=1e-5, rtol=1e-5)


def test_partial_combine_size_roundtrip():
    """(storage algebra, combine width) is the complete physical plan: the
    collective steps must be derivable from the PartialFragment alone. The
    materializer ICHECKs this equivalence against the loop-layout-derived
    plan on every compiled kernel; here the node-side derivation is checked
    structurally on representative decompositions."""
    from tilelang import _ffi_api

    def steps(partial):
        return [tuple(map(int, s)) for s in _ffi_api.PartialFragment_combine_steps(partial)]

    # Wide / FullParticipant: every replica is an addend lane.
    wide = tilelang.layout.make_fully_replicated_partial_fragment([64], 128)
    assert int(wide.combine_size) == 128
    assert steps(wide) == [(128, 1)]

    # Middle plan, lanes at thread stride 1.
    low = T.PartialFragment((32,), forward_thread_fn=lambda i, rep: (i % 16) * 8 + rep, replicate=8)
    assert int(low.combine_size) == 8
    assert steps(low) == [(8, 1)]

    # Same lane count at thread stride 16: a different communication
    # pattern the width alone could not distinguish.
    high = T.PartialFragment((32,), forward_thread_fn=lambda i, rep: rep * 16 + i % 16, replicate=8)
    assert int(high.combine_size) == 8
    assert steps(high) == [(128, 16)]


def test_same_storage_different_combine_widens_at_commit():
    """Two update sites over a scalar reducer induce the SAME storage
    algebra (thread = _rep, 128 partials) but DIFFERENT combine
    decompositions (8 vs 16 addend lanes). The combine width is part of the
    layout's identity, so the commit point widens to the FullParticipant
    plan instead of silently unifying them, and the numerics stay exact."""
    threads = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((16,), T.float32),
        C: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            a_frag = T.alloc_fragment((8,), T.float32)
            b_frag = T.alloc_fragment((16,), T.float32)
            T.copy(A, a_frag)
            T.copy(B, b_frag)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(8):
                T.reducer_update(acc[0], a_frag[i])
            for j in T.Parallel(16):
                T.reducer_update(acc[0], b_frag[j])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                C[0] = result[0]

    kern = tl.compile(kernel, out_idx=-1)
    assert f"SumOp, {threads}" in kern.get_kernel_source(), kern.get_kernel_source()
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    C = kern(A, B)
    torch.testing.assert_close(C, (A.sum() + B.sum()).reshape(1), atol=0, rtol=0)


def test_partial_fragment_combine_must_divide_replicate():
    """The low-bits convention `_rep = lane + combine * copy` only makes
    sense when `combine` evenly divides `replicate`."""
    with pytest.raises(ValueError, match="divide"):
        T.PartialFragment((32,), forward_thread_fn=lambda i, rep: rep, replicate=8, combine=3)


def test_partial_fragment_explicit_combine_construction():
    """`combine=` declares a decomposition with copy groups: only the low
    `combine` lanes are collective addends, the high `replicate / combine`
    groups are equal-value copies the finalize never reduces over."""
    from tilelang import _ffi_api

    pf = T.PartialFragment((1,), forward_thread_fn=lambda i, rep: rep, replicate=128, combine=8)
    assert int(pf.combine_size) == 8
    assert "(combine 8 x copy 16)" in repr(pf), repr(pf)
    steps = [tuple(map(int, s)) for s in _ffi_api.PartialFragment_combine_steps(pf)]
    assert steps == [(8, 1)]  # lanes only; the 16 copy groups need no communication


def _scalar_sum_copy_groups_kernel(combine: int, extent=8, threads=128):
    """Scalar sum whose update loop necessarily carries replication
    (Parallel(extent) on threads > extent): only expressible as an
    annotation since `combine=` exists — the total replication is pinned to
    the participant count with `combine` addend lanes in the low bits."""

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.annotate_layout(
                {
                    acc: T.PartialFragment(
                        (1,),
                        forward_thread_fn=lambda i, rep: rep,
                        replicate=threads,
                        combine=combine,
                    )
                }
            )
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    return kernel


def test_annotated_partial_copy_groups_pins_replicated_loop():
    """Pinning the natural plan of a replicated update loop: replicate=128
    total partials, combine=8 addend lanes (16 copy groups). The finalize
    reduces exactly 8 wide; the width is forced by the strict annotation,
    so the assertion is architecture-independent."""
    extent, threads = 8, 128
    kern = tl.compile(_scalar_sum_copy_groups_kernel(combine=extent), out_idx=-1)
    src = kern.get_kernel_source()
    assert "AllReduce<tl::SumOp, 8," in src, src
    assert f"SumOp, {threads}" not in src, src
    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A.sum().reshape(1), atol=0, rtol=0)


def test_annotated_partial_copy_groups_middle_plan():
    """combine=4 (< the loop's natural 8): each lane locally accumulates 2
    consecutive elements, 32 copy groups, and finalize emits a width-4
    collective — a middle plan no automatic solution would pick."""
    extent = 8
    kern = tl.compile(_scalar_sum_copy_groups_kernel(combine=4), out_idx=-1)
    src = kern.get_kernel_source()
    assert "AllReduce<tl::SumOp, 4," in src, src
    assert "SumOp, 8," not in src, src
    A = torch.arange(1, extent + 1, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kern(A), A.sum().reshape(1), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
