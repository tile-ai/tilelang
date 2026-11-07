"""SM100 2-CTA TMA loads with a cluster barrier (#3395).

A ``cta_group::2`` load routes the bytes of both CTAs of a pair to the even CTA's
barrier, so each pair leader must expect ``2 * tile_bytes`` regardless of the
total cluster size. Expecting ``cluster_size * tile_bytes`` on rank 0 only
faults or hangs on clusters of 4.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

TILE = 64
TILE_BYTES = TILE * TILE * 2  # bfloat16
NUM_CLUSTERS = 4


def make_kernel(cluster_dims):
    cx, cy = cluster_dims
    ncta = cx * cy
    grid_x = NUM_CLUSTERS * cx
    M = NUM_CLUSTERS * ncta * TILE

    @T.prim_func
    def main(
        A: T.Tensor((M, 2 * TILE), T.bfloat16),
        Out: T.Tensor((M, TILE), T.bfloat16),
    ):
        with T.ClusterKernel(grid_x, cy, threads=128, cluster_dims=(cx, cy, 1)) as (bx, by):
            A_shared = T.alloc_shared((TILE, TILE), T.bfloat16)
            # Both warps 0 of a CTA pair arrive on the pair leader's barrier.
            loaded = T.alloc_cluster_barrier([64])
            tx = T.get_thread_binding()
            rank = T.block_rank_in_cluster()
            row = (by * grid_x + bx) * TILE
            if tx < 32:
                T.tma_copy(A[row : row + TILE, 0:TILE], A_shared, barrier=loaded[0])
                T.mbarrier_arrive(loaded[0], rank - rank % 2)
            if rank % 2 == 0:
                T.mbarrier_wait_parity(loaded[0], 0)
                T.copy(A_shared, Out[row : row + TILE, 0:TILE])
            T.cluster_sync()

    return main, M


CLUSTER_DIMS = [(2, 1), (1, 2), (4, 1), (2, 2)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("cluster_dims", CLUSTER_DIMS)
def test_2cta_cluster_expect_tx_codegen(cluster_dims):
    func, _ = make_kernel(cluster_dims)
    src = tilelang.compile(func, target={"kind": "cuda", "arch": "sm_100a"}).get_kernel_source()
    assert "tl::tma_load_2sm" in src
    expects = re.findall(r"expect_transaction\((\d+)\)", src)
    assert expects == [str(2 * TILE_BYTES)], expects
    # Every pair leader (even rank) expects its pair's bytes, not only rank 0.
    assert "tl::block_rank_in_cluster() % 2) == 0" in src, src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("cluster_dims", CLUSTER_DIMS)
def test_2cta_cluster_expect_tx_runtime(cluster_dims):
    func, M = make_kernel(cluster_dims)
    kernel = tilelang.compile(func)
    a = torch.randn(M, 2 * TILE, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, TILE, dtype=torch.bfloat16, device="cuda")
    kernel(a, out)
    torch.cuda.synchronize()

    cx, cy = cluster_dims
    grid_x = NUM_CLUSTERS * cx
    for by in range(cy):
        for bx in range(grid_x):
            rank = bx % cx + (by % cy) * cx
            rows = slice((by * grid_x + bx) * TILE, (by * grid_x + bx + 1) * TILE)
            if rank % 2 == 0:
                torch.testing.assert_close(out[rows], a[rows, :TILE], rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
