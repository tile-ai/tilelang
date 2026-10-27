"""L1 MX GEMM preserves scale staging while reusing its mutex buffers."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def _mx_l1_kernel(k):
    m, n = 64, 64
    sf_pairs = k // 64

    @T.prim_func
    def kernel(
        A: T.Tensor((m, k), "float8_e4m3fn"),
        B: T.Tensor((n, k), "float8_e4m3fn"),
        SA: T.Tensor((sf_pairs, m), "uint16"),
        SB: T.Tensor((sf_pairs, n), "uint16"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((m, k), "float8_e4m3fn")
            b = T.alloc_l1((n, k), "float8_e4m3fn")
            sa = T.alloc_l1((m, sf_pairs), "uint16")
            sb = T.alloc_l1((n, sf_pairs), "uint16")
            c = T.alloc_l0c((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a, b, c, sa, sb, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("k", [128, 1536], ids=["single-sub-k", "three-sub-k"])
def test_mx_l1_gemm_nonuniform_scales(target, k):
    generator = torch.Generator().manual_seed(29)
    a = (torch.randint(-4, 5, (64, k), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (64, k), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    # Each byte is an E8M0 exponent for one K32 group. Neighboring groups
    # and rows differ, so a stale scale stage cannot match the reference.
    sa = torch.randint(124, 131, (64, k // 32), generator=generator, dtype=torch.uint8)
    sb = torch.randint(124, 131, (64, k // 32), generator=generator, dtype=torch.uint8)
    scaled_a = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    scaled_b = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    expected = scaled_a @ scaled_b.T
    compiled = tilelang.compile(_mx_l1_kernel(k), target=target, out_idx=-1)
    actual = compiled(a.npu(), b.npu(), sa.view(torch.uint16).T.contiguous().npu(), sb.view(torch.uint16).T.contiguous().npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
