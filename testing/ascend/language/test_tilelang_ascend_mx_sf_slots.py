"""MX scale-factor slot semantics on the L0A/L0B shadow register file.

Hardware findings pinned by this test (probed 2026-09-18 on Ascend 950DT):

- The SF destination is strictly the data tile's L0 address in 16-byte units
  (``asc_copy_l12l0a_mx`` dst = ``(uintptr_t)data_ptr / 16``); offsetting it
  by one fractal produces wrong products.
- SF slots are STICKY: they survive overwrites of the data at the same L0
  address, so a block-scaled MAD after a plain data reload still applies the
  previously loaded scales. This makes hoisting scale loads out of data
  reload loops legal.
- The SF load and the data load are order-free on the MTE1 queue.

The kernel loads data+scales and MADs into C1, then overwrites the SAME L0A
tile with new data (no scale load) and MADs into C2. Stickiness means
``C2 == (x2 * sx) @ (w * sw)^T`` with the ORIGINAL scales.
"""

import re

import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

M, K, N = 32, 128, 32
DTYPE = "float8_e4m3fn"
SF_K = K // 64  # packed uint16 pairs, one pair per 64 K elements


@T.prim_func
def sf_sticky_kernel(
    X: T.Buffer((M, K), DTYPE),
    X2: T.Buffer((M, K), DTYPE),
    W: T.Buffer((N, K), DTYPE),
    SFX: T.Buffer((SF_K, M), "uint16"),
    SFW: T.Buffer((SF_K, N), "uint16"),
    C1: T.Buffer((M, N), "float32"),
    C2: T.Buffer((M, N), "float32"),
):
    with T.Kernel(1):
        res = T.alloc_l0c((M, N), "float32")
        x_l1 = T.alloc_l1((M, K), DTYPE)
        x2_l1 = T.alloc_l1((M, K), DTYPE)
        w_l1 = T.alloc_l1((N, K), DTYPE)
        xsf_l1 = T.alloc_l1((M, SF_K), "uint16")
        wsf_l1 = T.alloc_l1((N, SF_K), "uint16")
        x_l0a = T.alloc_l0a((M, K), DTYPE)
        w_l0b = T.alloc_l0b((N, K), DTYPE)
        x_l0a_sf = T.alloc_l0a_sf(x_l0a)
        w_l0b_sf = T.alloc_l0b_sf(w_l0b)
        # Stickiness is only observable when both MADs read the SAME L0
        # addresses: forbid ping-pong versioning of the L0 tiles.
        T.annotate_buffer_versions({x_l0a: 1, w_l0b: 1, res: 1})

        T.copy(X, x_l1)
        T.copy(X2, x2_l1)
        T.copy(W, w_l1)
        T.copy(SFX, xsf_l1, transpose=True)
        T.copy(SFW, wsf_l1, transpose=True)

        # Pass 1: data + scales, block-scaled MAD -> C1.
        T.copy(x_l1, x_l0a)
        T.copy(xsf_l1, x_l0a_sf)
        T.copy(w_l1, w_l0b)
        T.copy(wsf_l1, w_l0b_sf)
        T.gemm_blockscaled(x_l0a, w_l0b, res, x_l0a_sf, w_l0b_sf, transpose_B=True, clear_accum=True)
        T.copy(res, C1)

        # Pass 2: overwrite the SAME L0A tile with new data, NO scale load.
        T.copy(x2_l1, x_l0a)
        T.gemm_blockscaled(x_l0a, w_l0b, res, x_l0a_sf, w_l0b_sf, transpose_B=True, clear_accum=True)
        T.copy(res, C2)


def _rel(actual, expected):
    return ((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6)).item()


def test_mx_sf_slots_sticky_across_data_overwrite():
    compiled = tilelang.compile(sf_sticky_kernel, out_idx=None)

    # Premise: one SF load, two data loads into the same L0A tile, two MADs.
    source = compiled.get_kernel_source()
    assert len(re.findall(r"asc_copy_l12l0a_mx\(", source)) == 1, source
    assert len(re.findall(r"asc_copy_l12l0a\(", source)) == 2, source
    assert len(re.findall(r"asc_mmad_mx\(", source)) == 2, source

    torch.manual_seed(42)
    dev = torch.device("npu")
    x = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    x2 = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    sfx_e8m0 = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device=dev)
    sfw_e8m0 = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)

    c1 = torch.zeros(M, N, dtype=torch.float32, device=dev)
    c2 = torch.zeros(M, N, dtype=torch.float32, device=dev)
    compiled(
        x,
        x2,
        w,
        sfx_e8m0.view(torch.uint16).T.contiguous(),
        sfw_e8m0.view(torch.uint16).T.contiguous(),
        c1,
        c2,
    )
    torch.npu.synchronize()

    def scale(e8m0):
        return torch.pow(2.0, e8m0.float() - 127.0).repeat_interleave(32, dim=1)[:, :K]

    w_scaled_t = (w.float() * scale(sfw_e8m0)).T
    sx = scale(sfx_e8m0)
    assert _rel(c1, (x.float() * sx) @ w_scaled_t) < 1e-6
    # The second MAD applied the ORIGINAL scales to the overwritten data.
    assert _rel(c2, (x2.float() * sx) @ w_scaled_t) < 1e-6
    # And distinguishably so: it is neither unscaled-A nor cleared slots.
    assert _rel(c2, x2.float() @ w_scaled_t) > 1e-2


if __name__ == "__main__":
    tilelang.testing.main()
