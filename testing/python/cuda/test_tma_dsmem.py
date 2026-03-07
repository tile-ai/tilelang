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
import numpy as np


@tilelang.jit(verbose=True, execution_backend="cython")
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
            s_barrier = T.alloc_shared((1,), "uint64")

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)

            # Every CTA initialises its own barrier: expect 1 arrival
            # carrying N*4 bytes (the cp.async.bulk signals on completion).
            if T.get_thread_binding() == 0:
                T.mbarrier_init(s_barrier[0], 1)

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


def main():
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Compute Capability 9.0+, found {major}.{minor}")
        return

    N = 128
    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = torch.zeros(N, dtype=torch.float32, device="cuda")

    kernel = make_store_cluster_kernel(N)
    kernel(A, B)

    result = B.cpu().numpy()
    expected = A.cpu().numpy()

    print("Result  (first 8):", result[:8])
    print("Expected(first 8):", expected[:8])

    if np.allclose(result, expected):
        print("PASS: tma_store_cluster copy successful")
    else:
        diff = np.abs(result - expected).max()
        print(f"FAIL: max diff = {diff}")


if __name__ == "__main__":
    main()
