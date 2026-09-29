"""MX scales survive data reloads and follow their data allocation's owner."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


M = N = 16
K = 128
OUTER, INNER = 2, 3


def _scale_lifetime_kernel(placement):
    @T.prim_func
    def main(
        X: T.Tensor((M, K * INNER), "float8_e4m3fn"),
        W: T.Tensor((N, K), "float8_e4m3fn"),
        SX: T.Tensor((2 * OUTER, M), "uint16"),
        SW: T.Tensor((2, N), "uint16"),
        O: T.Tensor((OUTER, INNER, M, N), "float32"),
    ):
        with T.Kernel(1):
            x1 = T.alloc_l1((M, K * INNER), "float8_e4m3fn")
            w1 = T.alloc_l1((N, K), "float8_e4m3fn")
            sx1 = T.alloc_l1((M, 2 * OUTER), "uint16")
            sw1 = T.alloc_l1((N, 2), "uint16")
            x0 = T.alloc_l0a((M, K), "float8_e4m3fn")
            w0 = T.alloc_l0b((N, K), "float8_e4m3fn")
            sx0 = T.alloc_l0a_sf(x0)
            sw0 = T.alloc_l0b_sf(w0)
            acc = T.alloc_l0c((M, N), "float32")
            if placement == "pinned":
                T.annotate_buffer_versions({sx0: 1})
            T.copy(X, x1)
            T.copy(W, w1)
            T.copy(SX, sx1, transpose=True)
            T.copy(SW, sw1, transpose=True)
            T.copy(w1, w0)
            T.copy(sw1, sw0)
            if placement == "kernel":
                T.copy(sx1[:, :2], sx0)
            for j in T.Pipelined(OUTER, num_stages=2):
                if placement == "outer" or placement == "pinned":
                    T.copy(sx1[:, 2 * j : 2 * (j + 1)], sx0)
                for i in T.Pipelined(INNER, num_stages=2):
                    T.copy(x1[:, i * K : (i + 1) * K], x0)
                    if placement == "tile":
                        T.copy(sx1[:, 2 * j : 2 * (j + 1)], sx0)
                    T.gemm_blockscaled(x0, w0, acc, sx0, sw0, transpose_B=True, clear_accum=True)
                    T.copy(acc, O[j, i, :, :])

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("placement", ["tile", "outer", "kernel", "pinned"])
def test_scale_state_survives_data_reloads(placement):
    # Every output observes a distinct data reload. CPU reference avoids using
    # another NPU matmul to define the expected result.
    generator = torch.Generator().manual_seed(3)
    x = (torch.randint(-4, 5, (M, K * INNER), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    w = (torch.randint(-4, 5, (N, K), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    sx = torch.randint(124, 131, (M, 4 * OUTER), dtype=torch.uint8, generator=generator)
    sw = torch.randint(124, 131, (N, 4), dtype=torch.uint8, generator=generator)
    compiled = tilelang.compile(_scale_lifetime_kernel(placement), target="ascend", out_idx=-1)
    actual = compiled(x.npu(), w.npu(), sx.view(torch.uint16).T.contiguous().npu(), sw.view(torch.uint16).T.contiguous().npu()).cpu()

    def scale(e8m0):
        return torch.pow(2.0, e8m0.float() - 127).repeat_interleave(32, dim=1)

    expected = torch.empty_like(actual)
    b = w.float() * scale(sw)
    for j in range(OUTER):
        sf_index = 0 if placement == "kernel" else j
        a_scale = scale(sx[:, 4 * sf_index : 4 * (sf_index + 1)])
        for i in range(INNER):
            expected[j, i] = (x[:, i * K : (i + 1) * K].float() * a_scale) @ b.T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
