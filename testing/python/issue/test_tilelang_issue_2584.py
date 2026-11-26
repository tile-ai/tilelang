"""Regression test for GitHub issue #2584.

An aligned 1-D load keeps TMA. A provably misaligned global slice is rejected
when TMA is requested, and a plain T.copy store falls back.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

N, M = 1024, 512


def _load_kernel(global_off):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), Out: T.Tensor((M,), "float32")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((M,), "float32")
            T.copy(A[global_off : global_off + M], s, prefer_instruction="tma")
            for i in T.Parallel(M):
                Out[i] = s[i]

    return main


def _store_kernel(global_off, prefer_tma=False):
    kwargs = {"prefer_instruction": "tma"} if prefer_tma else {}

    @T.prim_func
    def main(A: T.Tensor((M,), "float32"), Out: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((M,), "float32")
            T.copy(A, s)
            T.copy(s, Out[global_off : global_off + M], **kwargs)

    return main


def _lower_sm90(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        return tilelang.lower(
            func,
            target=target,
            enable_device_compile=False,
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_aligned_1d_load_keeps_tma():
    kernel = tilelang.compile(_load_kernel(4), target="cuda")
    assert "tl::tma_load(" in kernel.get_kernel_source()
    a = torch.arange(1, N + 1, device="cuda", dtype=torch.float32)
    out = torch.zeros(M, device="cuda", dtype=torch.float32)
    kernel(a, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a[4 : 4 + M], rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_misaligned_1d_load_reports():
    with pytest.raises(
        tvm.error.InternalError,
        match=r"\bA\b.*\b20\b.*16-byte",
    ):
        _lower_sm90(_load_kernel(5))


@tilelang.testing.requires_cuda
def test_misaligned_1d_store_reports():
    with pytest.raises(
        tvm.error.InternalError,
        match=r"\bOut\b.*\b20\b.*16-byte",
    ):
        _lower_sm90(_store_kernel(5, prefer_tma=True))


@tilelang.testing.requires_cuda
def test_misaligned_1d_store_falls_back():
    artifact = _lower_sm90(_store_kernel(5))
    assert "tl::tma_store(" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
