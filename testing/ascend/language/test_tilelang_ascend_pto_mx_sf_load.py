"""PTO source lowering for standalone MX scale-factor loads."""

import re

import tilelang
import tilelang.ascend.language as T


def _standalone_mx_sf_kernel():
    m, k, n = 32, 128, 32
    sf_k = k // 64

    @T.prim_func
    def kernel(
        x: T.Buffer((m, k), "float8_e4m3fn"),
        w: T.Buffer((n, k), "float8_e4m3fn"),
        sfx: T.Buffer((sf_k, m), "uint16"),
        sfw: T.Buffer((sf_k, n), "uint16"),
        out: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((m, k), "float8_e4m3fn")
            w_l1 = T.alloc_l1((n, k), "float8_e4m3fn")
            sfx_l1 = T.alloc_l1((m, sf_k), "uint16")
            sfw_l1 = T.alloc_l1((n, sf_k), "uint16")
            x_l0 = T.alloc_l0a((m, k), "float8_e4m3fn")
            w_l0 = T.alloc_l0b((n, k), "float8_e4m3fn")
            x_l0_sf = T.alloc_l0a_sf(x_l0)
            w_l0_sf = T.alloc_l0b_sf(w_l0)
            acc = T.alloc_l0c((m, n), "float32")

            T.copy(x, x_l1)
            T.copy(w, w_l1)
            T.copy(sfx, sfx_l1, transpose=True)
            T.copy(sfw, sfw_l1, transpose=True)
            T.copy(x_l1, x_l0)
            T.copy(sfx_l1, x_l0_sf)
            T.copy(w_l1, w_l0)
            T.copy(sfw_l1, w_l0_sf)
            T.gemm_blockscaled(
                x_l0,
                w_l0,
                acc,
                x_l0_sf,
                w_l0_sf,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(acc, out)

    return kernel


def test_pto_emits_standalone_mx_sf_loads():
    source = tilelang.lower(_standalone_mx_sf_kernel(), target="pto").kernel_source

    assert source.count("pto.mte_l1_l0a_mx(") == 1, source
    assert source.count("pto.mte_l1_l0b_mx(") == 1, source
    assert not re.search(r"tl\.ascend_load_c[ab]_sf", source), source
