"""T.reduce within one wavefront needs no LDS workspace, and so no barriers around it."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch
from tilelang.tileop.base import GemmWarpPolicy


@tilelang.jit(out_idx=[-1], target="hip")
def _row_sum(M, N, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, 32), T.float16),
        B: T.Tensor((N, 32), T.float16),
        out: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, 32), T.float16)
            B_s = T.alloc_shared((N, 32), T.float16)
            s = T.alloc_fragment((M, N), T.float32)
            r = T.alloc_fragment((M,), T.float32)
            T.copy(A[0, 0], A_s)
            T.copy(B[0, 0], B_s)
            T.clear(s)
            T.gemm(A_s, B_s, s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
            T.reduce_sum(s, r, dim=1)
            for i in T.Parallel(M):
                out[i] = r[i]

    return main


# FullRow: one wavefront holds a row, so with several wavefronts the reduce stays inside one.
# (threads, rows): 256 threads and 64 rows give one wavefront per 16 rows. A single 64-thread block is also in-wave.
@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("threads", "M", "N"), [(64, 64, 64), (256, 128, 64), (256, 64, 64)])
def test_in_wave_reduce_has_no_workspace(threads, M, N):
    kernel = _row_sum(M, N, threads)
    assert "workspace" not in kernel.get_kernel_source()
    A = torch.randn(M, 32, dtype=torch.float16, device="cuda")
    B = torch.randn(N, 32, dtype=torch.float16, device="cuda")
    out = kernel(A, B)
    ref = (A.float() @ B.float().T).sum(dim=1)
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


if __name__ == "__main__":
    tilelang.testing.main()
