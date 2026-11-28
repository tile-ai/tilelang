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
    x = torch.tensor([0.7, 0.7, 0.7, 0.7, -2.0, -2.0, -2.0, -2.0, -2.0], dtype=torch.float32)
    y = torch.tensor([-248.625, -249.0, 291.0, 292.0, 2.0, 3.0, -149.0, -151.0, 129.0], dtype=torch.float32)
    # Frozen A5 failures include both sides of overflow and a rounded-away least subnormal.
    x_bits = [0x3F378339, 0x3F3156AD, 0x3F16A4CB, 0x77EA90CC, 0x3F7FACFB, 0x3FD33766]
    y_bits = [0xC3854281, 0xC371ACBE, 0xC32751CE, 0x3F912738, 0x47A03405, 0x433124BC]
    # For x = 2 the input bound is 2 * abs(y). Check both neighbors of y = 63.
    x_bits.extend([0x40000000] * 3)
    y_bits.extend([0x427BFFFF, 0x427C0000, 0x427C0001])
    x = torch.cat((x, torch.tensor(x_bits, dtype=torch.uint32).view(torch.float32)))
    y = torch.cat((y, torch.tensor(y_bits, dtype=torch.uint32).view(torch.float32)))
    actual = pow_kernel(len(x))(x.npu(), y.npu()).cpu()
    expected = x.double().pow(y.double()).float()
    # A zero, an infinity or the least subnormal must match exactly. For other
    # finite values retain the same four-ULP limit as the three-SDK sweep.
    bits = expected.view(torch.int32)
    exact = torch.isinf(expected) | ((bits & 0x7FFFFFFF) <= 1)
    assert torch.equal(actual.view(torch.int32)[exact], bits[exact])
    assert torch.isfinite(actual[~exact]).all()
    delta = actual.view(torch.int32).long() - bits.long()
    assert (delta[~exact].abs() <= 4).all(), (actual, expected)
