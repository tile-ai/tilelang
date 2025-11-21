import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def get_kernel():
    @T.prim_func
    def main(
            x: T.Tensor((32), "float32")

    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            local_val = T.alloc_local([1], "float32")
            local_val[0] = x[tx]
            reduced_val = T.warp_reduce_sum(local_val[0])
            x[tx] = reduced_val
    return main


if __name__ == '__main__':
    a = torch.randn((32,), dtype=torch.float32, device='cuda')
    kernel = get_kernel()
    print(kernel.get_kernel_source())
    ref = torch.full_like(a, a.sum())
    kernel(a)
    torch.testing.assert_close(a, ref)
    print('Test passed for warp reduce sum ✅')

