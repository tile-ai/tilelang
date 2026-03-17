"""
Regression tests for SM-to-SM cluster copy (T.copy_cluster with dst_block).

Three lowering paths are covered:

  Fast path  (test_tma_store_cluster):
    T.copy_cluster(src, dst, dst_block=1, remote_barrier=bar)
    → single tl::tma_store_cluster issued by one thread; mbarrier completion
      is tracked by the TMA hardware via mbarrier.arrive.expect_tx.

  SIMT fallback, no barrier  (test_store_cluster_simt_no_barrier):
    T.copy_cluster(src, dst, dst_block=1)   # no remote_barrier
    → element-wise cooperative_groups::map_shared_rank stores by all threads;
      caller uses T.cluster_sync() for ordering.

  SIMT fallback, with barrier  (test_store_cluster_simt_barrier):
    T.copy_cluster(src2d[0:M, 0:N_tile], dst2d[0:M, 0:N_tile],
                   dst_block=1, remote_barrier=bar)
    where N_tile < N_full, so the inner-dim extent fails the contiguity check.
    → element-wise map_shared_rank stores, followed by auto-injected
        __syncthreads();
        if (threadIdx.x == 0) s_barrier[0].arrive(1u);
      so the destination CTA can wait on the same mbarrier as in the fast path.
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import numpy as np


# ---------------------------------------------------------------------------
# Fast path kernel
# ---------------------------------------------------------------------------


def make_store_cluster_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i in T.Parallel(N):
                    s_src[i] = A[i]
                T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=s_barrier[0])

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


# ---------------------------------------------------------------------------
# SIMT fallback, no barrier
# ---------------------------------------------------------------------------


def make_store_cluster_simt_no_barrier_kernel(N: int):
    """No remote_barrier → SIMT fallback always taken; cluster_sync() orders stores."""

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i in T.Parallel(N):
                    s_src[i] = A[i]
                # No remote_barrier: LowerClusterCopy always takes the SIMT path.
                # All threads write into block 1's s_dst via map_shared_rank.
                T.copy_cluster(s_src, s_dst, dst_block=1)

            # Full cluster barrier: ensures all map_shared_rank stores from
            # block 0 are visible in block 1's address space before block 1
            # reads s_dst.
            T.cluster_sync()

            if pid == 1:
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


# ---------------------------------------------------------------------------
# SIMT fallback, with auto-injected ptx_arrive_cluster_barrier
# ---------------------------------------------------------------------------


def make_store_cluster_simt_barrier_kernel(M: int, N_full: int, N_tile: int):
    """2-D slice copy that forces the SIMT fallback even though remote_barrier is set.

    s_src / s_dst are allocated with inner dimension N_full, but only the
    first N_tile columns are copied.  Because N_tile < N_full the
    is_contiguous_region() check fails: the inner-dim extent of the copy
    region (N_tile) does not equal the buffer shape (N_full).

    LowerClusterCopy falls back to map_shared_rank stores and, because
    remote_barrier was supplied, automatically appends:
        __syncthreads();
        if (threadIdx.x == 0) s_barrier[0].arrive(1u);
    Block 1 therefore waits on the same mbarrier as in the fast-path API,
    verifying that ptx_arrive_cluster_barrier is injected and functional.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N_tile), "float32"),
        B: T.Tensor((M, N_tile), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            # Deliberately wider buffer: N_full > N_tile so the slice
            # [0:M, 0:N_tile] is non-contiguous in row-major storage.
            s_src = T.alloc_shared((M, N_full), "float32")
            s_dst = T.alloc_shared((M, N_full), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i, j in T.Parallel(M, N_tile):
                    s_src[i, j] = A[i, j]

                # [0:M, 0:N_tile] inner-dim extent N_tile != N_full
                # → contiguity check fails → SIMT fallback.
                # Compiler auto-injects: __syncthreads() +
                #   if (t == 0) s_barrier[0].arrive(1u);
                T.copy_cluster(
                    s_src[0:M, 0:N_tile],
                    s_dst[0:M, 0:N_tile],
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                # Block 1 waits on the auto-injected ptx_arrive_cluster_barrier.
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for i, j in T.Parallel(M, N_tile):
                    B[i, j] = s_dst[i, j]

    return kernel


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster():
    """Fast path: T.copy_cluster emits tl::tma_store_cluster."""
    N = 128
    prim_func = make_store_cluster_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "tl::tma_store_cluster" in src, (
        "Expected tl::tma_store_cluster in generated kernel source; "
        "T.copy_cluster(dst_block=..., remote_barrier=...) may have regressed "
        f"to the SIMT fallback.\nKernel source:\n{src}"
    )

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="tma_store_cluster copy produced wrong result",
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_store_cluster_simt_no_barrier():
    """SIMT fallback (no remote_barrier): map_shared_rank + cluster_sync ordering."""
    N = 128
    prim_func = make_store_cluster_simt_no_barrier_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "map_shared_rank" in src, f"Expected map_shared_rank in generated source for no-barrier SIMT fallback.\nKernel source:\n{src}"
    assert "tl::tma_store_cluster" not in src, f"No-barrier path must NOT emit tl::tma_store_cluster.\nKernel source:\n{src}"

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="SIMT no-barrier cluster copy produced wrong result",
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_store_cluster_simt_barrier():
    """SIMT fallback with auto-injected ptx_arrive_cluster_barrier.

    A non-full-span 2-D slice forces the fallback even though remote_barrier
    is supplied.  The auto-injected arrive lets block 1 wait on the same
    mbarrier as in the fast-path API, verifying barrier correctness.
    """
    M, N_full, N_tile = 4, 64, 32  # M * N_tile == 128 == thread count

    prim_func = make_store_cluster_simt_barrier_kernel(M, N_full, N_tile)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "map_shared_rank" in src, f"Expected map_shared_rank for SIMT+barrier fallback.\nKernel source:\n{src}"
    assert "tl::tma_store_cluster" not in src, f"Non-contiguous 2-D slice must NOT emit tl::tma_store_cluster.\nKernel source:\n{src}"

    A = torch.arange(M * N_tile, dtype=torch.float32, device="cuda").reshape(M, N_tile)
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="SIMT+auto-barrier cluster copy produced wrong result",
    )


if __name__ == "__main__":
    tilelang.testing.main()
