"""MX scale-factor slot planning: lockstep rings and hoisted broadcasts.

From PR #475 review: the scale slots are keyed to the data tile's address, so
a multi-buffered tile splits its slot state per version. A scale load ringing
in the same pipelined loop as its tile fills each stage's slot in lockstep; a
scale load hoisted outside that loop (into an outer pipelined loop, or above
every loop) is single-version and broadcast into every version slot of the
tile — its scales are invariant across the tile's ring period, slots are
sticky, and the plain storage dependence on the handle orders each broadcast
against every consumer of the previous scales.
"""

import torch
import tilelang
from tilelang.ascend import language as T

M = N = 32
K = 128
OUTER = INNER = 4


def _hoisted_sf_kernel(sf_in_outer_loop, pin_single_version=False):
    @T.prim_func
    def main(
        X: T.Buffer((M, K * INNER), "float8_e4m3fn"),
        W: T.Buffer((N, K), "float8_e4m3fn"),
        SX: T.Buffer((2 * OUTER, M), "uint16"),
        SW: T.Buffer((2, N), "uint16"),
        O: T.Buffer((M, N), "float32"),
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

            if pin_single_version:
                T.annotate_buffer_versions({x0: 1})

            T.copy(X, x1)
            T.copy(W, w1)
            T.copy(SX, sx1, transpose=True)
            T.copy(SW, sw1, transpose=True)
            T.copy(w1, w0)
            T.copy(sw1, sw0)

            if sf_in_outer_loop:
                for j in T.Pipelined(OUTER, num_stages=2):
                    T.copy(sx1[:, 2 * j : 2 * (j + 1)], sx0)
                    for i in T.Pipelined(INNER, num_stages=2):
                        T.copy(x1[:, i * K : (i + 1) * K], x0)
                        T.gemm_blockscaled(x0, w0, acc, sx0, sw0, transpose_B=True, clear_accum=(j == 0 and i == 0))
            else:
                T.copy(sx1[:, 0:2], sx0)
                for i in T.Pipelined(INNER, num_stages=2):
                    T.copy(x1[:, i * K : (i + 1) * K], x0)
                    T.gemm_blockscaled(x0, w0, acc, sx0, sw0, transpose_B=True, clear_accum=(i == 0))
            T.copy(acc, O)

    return main


def _lower(kernel):
    return tilelang.lower(kernel, target="ascend").kernel_source


def test_hoisted_sf_load_broadcasts_into_every_tile_slot():
    source = _lower(_hoisted_sf_kernel(sf_in_outer_loop=True))
    # The scale load in the outer loop writes both slots of the inner ring...
    assert source.count("asc_copy_l12l0a_mx(") == 2, source
    assert "(__ca__ fp8_e4_t*)x0)) / 16" in source, source
    assert "(__ca__ fp8_e4_t*)x0 + 4096)) / 16" in source, source
    # ...while the data tile and the MAD keep their ping-pong untouched.
    assert "(i & 1) * 4096" in source, source


def test_fully_hoisted_sf_load_broadcasts():
    source = _lower(_hoisted_sf_kernel(sf_in_outer_loop=False))
    assert source.count("asc_copy_l12l0a_mx(") == 2, source
    assert "(__ca__ fp8_e4_t*)x0 + 4096)) / 16" in source, source


def test_pinned_tile_keeps_a_single_slot():
    source = _lower(_hoisted_sf_kernel(sf_in_outer_loop=True, pin_single_version=True))
    assert source.count("asc_copy_l12l0a_mx(") == 1, source
    assert "x0 + 4096" not in source, source


def test_hoisted_sf_broadcast_numerics():
    torch.manual_seed(3)
    dev = torch.device("npu")
    x = (torch.randn(M, K * INNER, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    sx_e8m0 = torch.randint(124, 131, (M, 4 * OUTER), dtype=torch.uint8, device=dev)
    sw_e8m0 = torch.randint(124, 131, (N, 4), dtype=torch.uint8, device=dev)

    compiled = tilelang.compile(_hoisted_sf_kernel(sf_in_outer_loop=True), out_idx=-1)
    actual = compiled(
        x,
        w,
        sx_e8m0.view(torch.uint16).T.contiguous(),
        sw_e8m0.view(torch.uint16).T.contiguous(),
    )
    torch.npu.synchronize()

    def scale(e8m0):
        return torch.pow(2.0, e8m0.float() - 127.0).repeat_interleave(32, dim=1)

    b_scaled = w.float() * scale(sw_e8m0)
    expected = torch.zeros(M, N, device=dev)
    for j in range(OUTER):
        a_scale = scale(sx_e8m0[:, 4 * j : 4 * (j + 1)])
        for i in range(INNER):
            expected += (x[:, i * K : (i + 1) * K].float() * a_scale) @ b_scaled.T

    rel = ((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6)).item()
    assert rel < 1e-6, f"rel={rel:.4e}"


if __name__ == "__main__":
    tilelang.testing.main()
