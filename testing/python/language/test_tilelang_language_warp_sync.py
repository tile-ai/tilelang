import tilelang
import tilelang.language as T
import torch
from tvm import tir
import tilelang.testing


@tilelang.jit
def kernel_with_warp_sync():
    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                tir.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


@tilelang.testing.requires_cuda
def test_warp_sync():
    a = torch.empty((1), device="cuda", dtype=torch.int32)
    b = torch.empty((1), device="cuda", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1


@tilelang.jit
def kernel_with_shfl_sync():
    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(0xFFFFFFFF, val, 31)
            A[tx] = broadcast

    return main


@tilelang.testing.requires_cuda
def test_shfl_sync():
    a = torch.empty((32), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_sync()
    assert "__shfl_sync" in kernel.get_kernel_source()
    kernel(a)
    assert torch.all(a == 310)


if __name__ == "__main__":
    tilelang.testing.main()
