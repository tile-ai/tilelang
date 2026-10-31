"""Regression test for GitHub issue #2584.

#2584 - A 1-D bulk TMA copy must not start at a global byte offset that is not
16-byte aligned. The compiler rejects such a copy when TMA is requested instead
of emitting an illegal instruction; aligned copies keep TMA.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

N, M = 1024, 512


def _load_kernel(global_off, prefer):
    kwargs = {"prefer_instruction": "tma"} if prefer else {}

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), Out: T.Tensor((M,), "float32")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((M,), "float32")
            T.copy(A[global_off : global_off + M], s, **kwargs)
            for i in T.Parallel(M):
                Out[i] = s[i]

    return main


def _store_kernel(global_off):
    @T.prim_func
    def main(A: T.Tensor((M,), "float32"), Out: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((M,), "float32")
            T.copy(A, s)
            T.copy(s, Out[global_off : global_off + M], prefer_instruction="tma")

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_aligned_1d_load_keeps_tma():
    kernel = tilelang.compile(_load_kernel(4, True), target="cuda")
    assert "tl::tma_load(" in kernel.get_kernel_source()
    a = torch.arange(1, N + 1, device="cuda", dtype=torch.float32)
    out = torch.zeros(M, device="cuda", dtype=torch.float32)
    kernel(a, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a[4 : 4 + M], rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_misaligned_1d_load_reports():
    with pytest.raises(Exception, match="16-byte aligned"):
        tilelang.compile(_load_kernel(5, True), target="cuda")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_misaligned_1d_store_reports():
    with pytest.raises(Exception, match="16-byte aligned"):
        tilelang.compile(_store_kernel(5), target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
