import tilelang.language as T
import tilelang.testing
import torch


@tilelang.jit(out_idx=[-1])
def _ceildiv_kernel(a: int, b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


def run_ceildiv(a=128, b=32):
    kernel = _ceildiv_kernel(a, b)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)


def test_ceildiv():
    run_ceildiv(a=128, b=32)
    run_ceildiv(a=1, b=32)
    run_ceildiv(a=-1, b=32)
    run_ceildiv(a=-2, b=32)


@tilelang.jit
def _ceildiv_kernel_dyn(b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32), a: T.int32):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


def run_ceildiv_dyn(a=128, b=32):
    kernel = _ceildiv_kernel_dyn(b)
    A = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(A, a)
    print(kernel.get_kernel_source())
    print(A)


@tilelang.testing.requires_cuda
def test_ceildiv_dyn():
    run_ceildiv_dyn(a=128, b=32)
    run_ceildiv_dyn(a=1, b=32)
    run_ceildiv_dyn(a=-1, b=32)
    run_ceildiv_dyn(a=-2, b=32)


if __name__ == "__main__":
    tilelang.testing.main()
