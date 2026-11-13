"""Regression for tile-ai/tilelang#2525: a dynamic 1-D shared->global copy
must not select a descriptor-based bulk TMA store, which cannot encode a
symbolic tile extent and truncates an unaligned runtime tail.
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

NMAX = 64


def _kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=128):
            sh = T.alloc_shared((NMAX,), "float32")
            m = T.min(n, NMAX)
            for i in T.Parallel(NMAX):
                if i < n:
                    sh[i] = A[i]
            T.copy(sh[0:m], B[0:m])

    return main


@tilelang.testing.requires_cuda
def test_dynamic_extent_shared_to_global_copy_keeps_unaligned_tail():
    kernel = tilelang.compile(_kernel(), target="cuda")

    for n in (63, 64):
        a = torch.arange(1, n + 1, dtype=torch.float32, device="cuda")
        b = torch.zeros_like(a)
        kernel(a, b)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"dynamic extent n={n} lost data; tail={b[-4:].tolist()}, expected={a[-4:].tolist()}"


if __name__ == "__main__":
    tilelang.testing.main()
