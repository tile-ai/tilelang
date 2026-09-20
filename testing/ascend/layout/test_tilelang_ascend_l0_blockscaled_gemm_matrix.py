"""Packed MX scales and transpose modes preserve the logical FP8 matrix product."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


def _kernel(m, n, trans_a, trans_b):
    k = 128
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def gemm(
        a: T.Tensor(a_shape, "float8_e4m3fn"),
        b: T.Tensor(b_shape, "float8_e4m3fn"),
        sa: T.Tensor((2, m), "uint16"),
        sb: T.Tensor((2, n), "uint16"),
        out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_shape, "float8_e4m3fn")
            b_l1 = T.alloc_l1(b_shape, "float8_e4m3fn")
            sa_l1 = T.alloc_l1((m, 2), "uint16")
            sb_l1 = T.alloc_l1((n, 2), "uint16")
            a_l0 = T.alloc_l0a(a_shape, "float8_e4m3fn")
            b_l0 = T.alloc_l0b(b_shape, "float8_e4m3fn")
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.copy(sa, sa_l1, transpose=True)
            T.copy(sb, sb_l1, transpose=True)
            T.copy(a_l1, a_l0, scale=sa_l1)
            T.copy(b_l1, b_l0, scale=sb_l1)
            T.blockscaled_gemm(a_l0, b_l0, acc, transpose_A=trans_a, transpose_B=trans_b, clear_accum=True)
            T.copy(acc, out)

    return gemm


@pytest.mark.parametrize(
    "m,n,trans_a,trans_b",
    [
        (1, 15, False, True),
        (15, 15, False, True),
        (32, 32, False, True),
        (100, 100, False, True),
        (64, 64, False, False),
        (64, 64, True, False),
        (64, 64, True, True),
    ],
)
def test_blockscaled_l0_gemm(m, n, trans_a, trans_b):
    # Small binary fractions keep FP8 input conversion exact; vary every K32 scale group.
    a = (torch.randint(-4, 5, (m, 128), device="npu").float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, 128), device="npu").float() / 8).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, 4), dtype=torch.uint8, device="npu")
    sb = torch.randint(124, 131, (n, 4), dtype=torch.uint8, device="npu")
    expected = (a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)) @ (
        b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ).T
    kernel = tilelang.compile(_kernel(m, n, trans_a, trans_b), target="ascend", out_idx=-1)
    actual = kernel(
        a.T.contiguous() if trans_a else a,
        b if trans_b else b.T.contiguous(),
        sa.view(torch.uint16).T.contiguous(),
        sb.view(torch.uint16).T.contiguous(),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
