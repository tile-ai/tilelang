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
            acc = T.alloc_reducer((1,), T.float32, op="sum", seed=seed)
            T.reducer_init(acc)
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


def test_seed_falls_back_to_baseline():
    """The narrow plan does not support seeds yet; the epoch must fall back
    to the wide plan and stay numerically correct."""
    extent, threads, seed = 8, 128, 100.0

    @T.prim_func
    def kernel(A: T.Tensor((extent,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((extent,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum", seed=seed)
            T.reducer_init(acc)
            for i in T.Parallel(extent):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    source = tl.compile(kernel, out_idx=-1).get_kernel_source()
    assert f"SumOp, {threads}" in source, source


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
# Lifecycle / access diagnostics
# ---------------------------------------------------------------------------


def _compile_expect_error(kernel_factory, match):
    with pytest.raises(Exception, match=match):
        kernel = kernel_factory()
        tl.compile(kernel, out_idx=-1)


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


if __name__ == "__main__":
    tilelang.testing.main()
