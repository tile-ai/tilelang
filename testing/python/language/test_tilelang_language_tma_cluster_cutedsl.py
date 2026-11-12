"""CuTeDSL cluster copies: TMA multicast, L2 eviction hints, SM-to-SM bulk
copies and SIMT cluster stores, mirroring the CUDA coverage in
test_tma_multicast_demo.py and test_tma_dsmem.py."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _compile(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cutedsl")


def _multicast_kernel(M, N, block_M, block_N, cluster_mask, eviction_policy=None):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.ClusterKernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
            cluster_dims=(4, 1, 1),
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            T.copy_cluster(
                A[by * block_M, bx * block_N],
                A_shared,
                cluster_mask=cluster_mask,
                eviction_policy=eviction_policy,
            )
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return kernel


def _check_multicast(eviction_policy=None):
    M, N = 1024, 1024
    block_M, block_N = 128, 64
    kernel = _multicast_kernel(M, N, block_M, block_N, 0b0011, eviction_policy)
    mod = _compile(kernel, [1])

    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = mod(A)

    # Rank 1 receives the tile rank 0 loaded; ranks 2 and 3 load their own.
    torch.testing.assert_close(B[0:block_M, 0:block_N], A[0:block_M, 0:block_N], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        B[0:block_M, block_N : 2 * block_N],
        A[0:block_M, 0:block_N],
        rtol=1e-2,
        atol=1e-2,
    )
    for rank in (2, 3):
        lo, hi = rank * block_N, (rank + 1) * block_N
        torch.testing.assert_close(B[0:block_M, lo:hi], A[0:block_M, lo:hi], rtol=1e-2, atol=1e-2)


def _store_cluster_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.ClusterKernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
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


def _store_cluster_wide_kernel(M: int, N_full: int, N_tile: int):
    """A non-contiguous slice copy: SIMT stores into the peer CTA plus a barrier."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N_tile), "float32"),
        B: T.Tensor((M, N_tile), "float32"),
    ):
        with T.ClusterKernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((M, N_full), "float32")
            s_dst = T.alloc_shared((M, N_full), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i, j in T.Parallel(M, N_tile):
                    s_src[i, j] = A[i, j]
                T.copy_cluster(
                    s_src[0:M, 0:N_tile],
                    s_dst[0:M, 0:N_tile],
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for i, j in T.Parallel(M, N_tile):
                    B[i, j] = s_dst[i, j]

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_multicast_cutedsl():
    _require_cutedsl()
    _check_multicast()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_multicast_eviction_cutedsl():
    _require_cutedsl()
    _check_multicast(eviction_policy="evict_last")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster_cutedsl():
    _require_cutedsl()
    N = 1024
    mod = _compile(_store_cluster_kernel(N), [1])
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = mod(A)
    torch.testing.assert_close(B, A)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_store_cutedsl():
    _require_cutedsl()
    M, N_full, N_tile = 32, 64, 32
    mod = _compile(_store_cluster_wide_kernel(M, N_full, N_tile), [1])
    A = torch.randn(M, N_tile, device="cuda", dtype=torch.float32)
    B = mod(A)
    torch.testing.assert_close(B, A)


if __name__ == "__main__":
    tilelang.testing.main()
