"""TMA multicast validation."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def make_tma_multicast_demo_kernel(M, N, block_M, block_N, cluster_mask):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
            cluster_dims=(4, 1, 1),
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            T.copy_cluster(A[by * block_M, bx * block_N], A_shared, cluster_mask=cluster_mask)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_multicast_demo():
    """Verify TMA multicast: rank 1's B region should equal rank 0's A region within the same cluster."""
    M, N = 1024, 1024
    block_M, block_N = 128, 64
    # mask=0b0011: rank 0 multicasts, rank 1 receives, ranks 2/3 each do regular tma_load
    cluster_mask = 0b0011

    kernel = make_tma_multicast_demo_kernel(M, N, block_M, block_N, cluster_mask)
    mod = tilelang.compile(
        kernel,
        out_idx=[1],
        execution_backend="cython",
    )

    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = mod(A)

    # Within a cluster: the first 4 blocks in the grid are (0,0),(1,0),(2,0),(3,0) -> by=0, bx=0,1,2,3
    # rank 0 -> bx=0: A[0:block_M, 0:block_N] -> B[0:block_M, 0:block_N]
    # rank 1 -> bx=1: multicast receives A[0:block_M, 0:block_N] -> B[0:block_M, block_N:2*block_N]
    # rank 2 -> bx=2: A[0:block_M, 2*block_N:3*block_N] -> B[0:block_M, 2*block_N:3*block_N]
    # rank 3 -> bx=3: A[0:block_M, 3*block_N:4*block_N] -> B[0:block_M, 3*block_N:4*block_N]

    # Multicast check: rank 1's B region should equal rank 0's A region
    B_rank1 = B[0:block_M, block_N : 2 * block_N]
    A_rank0 = A[0:block_M, 0:block_N]
    torch.testing.assert_close(B_rank1, A_rank0, rtol=1e-2, atol=1e-2)

    # rank 0 itself: B should equal A
    torch.testing.assert_close(B[0:block_M, 0:block_N], A[0:block_M, 0:block_N], rtol=1e-2, atol=1e-2)
    # ranks 2, 3: each B region equals its own A region
    torch.testing.assert_close(
        B[0:block_M, 2 * block_N : 3 * block_N],
        A[0:block_M, 2 * block_N : 3 * block_N],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        B[0:block_M, 3 * block_N : 4 * block_N],
        A[0:block_M, 3 * block_N : 4 * block_N],
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
