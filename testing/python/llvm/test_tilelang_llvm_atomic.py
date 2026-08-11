"""LLVM atomic op tests (shares the CPU atomic impls with the `c` target).

Lightweight mirror of testing/python/cpu/test_tilelang_cpu_atomic.py covering
one representative case per lowering path, since `llvm` and `c` share the same
registries (TargetIsCPU matches kDLCPU) and the same LowerCPUAtomics pass:
- scalar add + return_prev (tl.transform.LowerCPUAtomics),
- tile-region max (src/cpu/op/atomic_reduce.cc),
- addx4 vector expansion (tl.transform.LowerCPUAtomics).
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _compile_llvm(func, out_idx):
    with tvm.target.Target("llvm"):
        return tilelang.compile(func, out_idx=out_idx, target="llvm", execution_backend="tvm_ffi")


@tilelang.testing.requires_llvm
def test_llvm_atomic_add_scalar_return_prev():
    N = 8
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        Init: T.Tensor((1,), dtype),
        B: T.Tensor((1,), dtype),
        P: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.serial(N):
                P[i] = T.atomic_add(B[0], A[i], return_prev=True)

    kernel = _compile_llvm(main, out_idx=[2, 3])
    gen = torch.Generator(device="cpu").manual_seed(7)
    A = torch.randn((N,), dtype=torch.float32, generator=gen)
    Init = torch.full((1,), 1.0, dtype=torch.float32)
    out_b, out_p = kernel(A, Init)
    expected_p = Init + torch.cumsum(A, dim=0) - A
    torch.testing.assert_close(out_p, expected_p, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_b, A.sum() + Init, rtol=1e-4, atol=1e-4)


@tilelang.testing.requires_llvm
def test_llvm_atomic_max_region():
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Init: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            for i, j in T.grid(M, N):
                B[i, j] = Init[i, j]
            T.atomic_max(B, A)

    kernel = _compile_llvm(main, out_idx=[2])
    gen = torch.Generator(device="cpu").manual_seed(7)
    A = torch.randn((M, N), dtype=torch.float32, generator=gen)
    Init = torch.randn((M, N), dtype=torch.float32, generator=gen)
    out = kernel(A, Init)
    torch.testing.assert_close(out, torch.maximum(Init, A), rtol=1e-4, atol=1e-4)


@tilelang.testing.requires_llvm
def test_llvm_atomic_addx4():
    dtype = "float32"

    @T.prim_func
    def main(Val: T.Tensor((4,), dtype), Init: T.Tensor((4,), dtype), Dst: T.Tensor((4,), dtype)):
        with T.Kernel(1):
            for i in T.serial(4):
                Dst[i] = Init[i]
            T.atomic_addx4(Dst[0:4], Val[0:4])

    kernel = _compile_llvm(main, out_idx=[2])
    gen = torch.Generator(device="cpu").manual_seed(7)
    Val = torch.randn((4,), dtype=torch.float32, generator=gen)
    Init = torch.randn((4,), dtype=torch.float32, generator=gen)
    out = kernel(Val, Init)
    torch.testing.assert_close(out, Init + Val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
