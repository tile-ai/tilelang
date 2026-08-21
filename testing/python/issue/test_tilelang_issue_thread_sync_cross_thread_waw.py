import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _copy_then_offset_fill(
    A: T.Tensor((16,), T.float32),
    B: T.Tensor((16,), T.float32),
):
    with T.Kernel(1, threads=16):
        a_shared = T.alloc_shared((16,), T.float32)
        T.copy(A, a_shared)
        T.fill(a_shared[10:16], 99.0)
        T.copy(a_shared, B)


@tilelang.testing.requires_cuda
def test_copy_then_offset_fill_orders_cross_thread_writes():
    a = torch.arange(16, device="cuda", dtype=torch.float32)
    expected = a.clone()
    expected[10:16] = 99.0

    for _ in range(10):
        b = torch.empty_like(a)
        _copy_then_offset_fill(a, b)
        torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
