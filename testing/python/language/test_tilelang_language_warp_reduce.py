import torch

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def get_kernel(reduce_op: str, dtype: str):
    assert reduce_op in ["sum", "max", "min", "bitand", "bitor"]

    @T.prim_func
    def main(x: T.Tensor((32), dtype)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            local_val = T.alloc_local([1], dtype)
            local_val[0] = x[tx]
            reduced_val = T.alloc_local([1], dtype)
            if reduce_op == "sum":
                reduced_val[0] = T.warp_reduce_sum(local_val[0])
            elif reduce_op == "max":
                reduced_val[0] = T.warp_reduce_max(local_val[0])
            elif reduce_op == "min":
                reduced_val[0] = T.warp_reduce_min(local_val[0])
            elif reduce_op == "bitand":
                reduced_val[0] = T.warp_reduce_bitand(local_val[0])
            elif reduce_op == "bitor":
                reduced_val[0] = T.warp_reduce_bitor(local_val[0])
            x[tx] = reduced_val[0]

    return main


def test_warp_reduce_sum():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("sum", T.float32)
    ref = torch.full_like(a, a.sum())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_max():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("max", T.float32)
    print(kernel.get_kernel_source())
    ref = torch.full_like(a, a.max())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_min():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("min", T.float32)
    ref = torch.full_like(a, a.min())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_bitand():
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32, device="cuda")
    kernel = get_kernel("bitand", T.int32)
    ref_val = a[0]
    for i in range(1, a.shape[0]):
        ref_val = ref_val & a[i]
    ref = torch.full_like(a, ref_val)
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_bitor():
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32, device="cuda")
    kernel = get_kernel("bitor", T.int32)
    ref_val = a[0]
    for i in range(1, a.shape[0]):
        ref_val = ref_val | a[i]
    ref = torch.full_like(a, ref_val)
    kernel(a)
    torch.testing.assert_close(a, ref)


if __name__ == "__main__":
    tilelang.testing.main()
