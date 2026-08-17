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

import pytest
import tilelang
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import torch

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
    (unchanged) projected collective."""
    M, K, threads = 32, 64, 128
    source = tl.compile(_rowsum_kernel(M, K, threads, T.float16), out_idx=-1).get_kernel_source()
    assert "_pk" in source, source
    assert "SumOp, 8" in source, source  # collective width unchanged by packing

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = tl.compile(_rowsum_kernel(M, K, threads, T.float16), out_idx=-1)(A)
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


if __name__ == "__main__":
    tilelang.testing.main()
