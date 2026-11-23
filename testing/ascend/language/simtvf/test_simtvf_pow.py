import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


@tilelang.jit(out_idx=[2], target="ascend")
def pow_kernel(n: int):
    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"), Out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128)) as bx, T.SimtVF(threads=128):
            i = bx * 128 + T.get_thread_binding()
            if i < n:
                Out[i] = T.pow(X[i], Y[i])

    return main


@tilelang.testing.requires_ascend
def test_simt_pow_fp32_range_boundaries():
    # Raw FP32 pairs reproduce the inputs on both sides of under/overflow.
    x = torch.tensor([0.7, 0.7, 0.7, 0.7, -2.0, -2.0], dtype=torch.float32)
    y = torch.tensor([-248.625, -249.0, 291.0, 292.0, 2.0, 3.0], dtype=torch.float32)
    x = torch.cat((x, torch.tensor([0x3F378339, 0x3F3156AD], dtype=torch.uint32).view(torch.float32)))
    y = torch.cat((y, torch.tensor([0xC3854281, 0xC371ACBE], dtype=torch.uint32).view(torch.float32)))
    actual = pow_kernel(len(x))(x.npu(), y.npu()).cpu()
    expected = x.double().pow(y.double()).float()
    # A zero, an infinity or the least subnormal must match exactly. For other
    # finite values retain the same four-ULP limit as the three-SDK sweep.
    bits = expected.view(torch.int32)
    exact = torch.isinf(expected) | (expected == 0) | (bits == 1)
    assert torch.equal(actual.view(torch.int32)[exact], bits[exact])
    assert torch.isfinite(actual[~exact]).all()
    delta = actual.view(torch.int32).long() - bits.long()
    assert (delta[~exact].abs() <= 4).all(), (actual, expected)
