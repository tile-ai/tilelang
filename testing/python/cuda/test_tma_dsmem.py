"""
Demo / regression test for SM-to-SM bulk async copy via tl::tma_store_cluster.

T.copy with dst_block + barrier now lowers to a single
tl::tma_store_cluster call instead of a SIMT element-by-element loop.

Expected generated producer code (block 0):
  if (((int)threadIdx.x) == 0) {
      tl::tma_store_cluster(&s_dst[0], &s_src[0], 1,
                            (uint32_t)(512), s_barrier[0]);
  }

Block 1 waits on its own s_barrier and then reads the result.
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import numpy as np


def make_store_cluster_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        # 2 CTAs in a cluster of size 2
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)

            T.cluster_sync()

            if pid == 0:
                # Load A into s_src
                for i in T.Parallel(N):
                    s_src[i] = A[i]

                # Bulk-async copy s_src (local) → s_dst (remote, block 1)
                # using tl::tma_store_cluster, signalling block 1's barrier.
                T.copy(s_src, s_dst, dst_block=1, remote_barrier=s_barrier[0])

            if pid == 1:
                # Wait until block 0 finishes writing to our s_dst.
                T.mbarrier_wait_parity(s_barrier[0], 0)

                # Store result to global memory
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster():

    N = 128
    prim_func = make_store_cluster_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    # Assert that the lowering actually produced tl::tma_store_cluster.
    # The SIMT fallback (map_shared_rank + scalar stores) also copies data
    # correctly, so a pure numerical check would miss a regression where
    # T.copy(dst_block=..., remote_barrier=...) stops emitting the bulk-async
    # cluster intrinsic.
    src = mod.get_kernel_source()
    assert "tl::tma_store_cluster" in src, (
        "Expected tl::tma_store_cluster in generated kernel source; "
        "T.copy(dst_block=..., remote_barrier=...) may have regressed to the "
        f"SIMT fallback.\nKernel source:\n{src}"
    )

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)

    result = B.cpu().numpy()
    expected = A.cpu().numpy()

    diff = np.abs(result - expected).max()
    assert np.allclose(result, expected), f"tma_store_cluster copy failed: max diff = {diff}"


if __name__ == "__main__":
    tilelang.testing.main()
