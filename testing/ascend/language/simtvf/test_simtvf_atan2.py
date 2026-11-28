import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


@tilelang.testing.requires_ascend
def test_atan2_rounds_tiny_angles_with_their_sign():
    # Use both neighbors of 2. At x=2, |y/x| is exactly halfway between
    # two FP32 values, and atan(|y/x|) lies just below that midpoint.
    x_bits = [0x3FFFFFFF, 0x40000000, 0x40000001] * 2
    y_bits = [1, 1, 1, 3, 3, 3]
    expected_bits = [1, 0, 0, 2, 1, 1]
    n = len(x_bits)

    @T.prim_func
    def main(Y: T.Tensor((n,), "float32"), X: T.Tensor((n,), "float32"), Out: T.Tensor((n,), "float32")):
        with T.Kernel(1), T.SimtVF(threads=32):
            i = T.get_thread_binding()
            if i < n:
                Out[i] = T.atan2(Y[i], X[i])

    kernel = tilelang.compile(main, out_idx=[2], target="ascend")
    x = torch.tensor(x_bits, dtype=torch.uint32).view(torch.float32).npu()
    for sign in (0, 0x80000000):
        y = torch.tensor([v | sign for v in y_bits], dtype=torch.uint32).view(torch.float32).npu()
        expected = torch.tensor([v | sign for v in expected_bits], dtype=torch.uint32)
        assert torch.equal(kernel(y, x).cpu().view(torch.uint32), expected)
