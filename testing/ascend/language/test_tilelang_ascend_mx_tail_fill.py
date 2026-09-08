"""Codegen regressions for automatic DeepGEMM-style MX K-tail fills."""

import re

import pytest
import tilelang
from tilelang.ascend import language as T


M = 64
N = 64
K_ALLOC = 128
SF_PAIRS = K_ALLOC // 64


def make_mx_actual_k_copy(major_mn: bool, actual_k: int, normalize_to_k_major: bool = False):
    a_shape = (K_ALLOC, M) if major_mn else (M, K_ALLOC)
    a_l1_shape = (M, K_ALLOC) if normalize_to_k_major else a_shape

    @T.prim_func
    def kernel(
        a: T.Buffer(a_shape, "float8_e4m3fn"),
        b: T.Buffer((N, K_ALLOC), "float8_e4m3fn"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_l1_shape, "float8_e4m3fn")
            b_l1 = T.alloc_l1((N, K_ALLOC), "float8_e4m3fn")
            sfa_l1 = T.alloc_l1((M, SF_PAIRS), "int16")
            sfb_l1 = T.alloc_l1((N, SF_PAIRS), "int16")
            a_l0 = T.alloc_l0a(a_l1_shape, "float8_e4m3fn")
            b_l0 = T.alloc_l0b((N, K_ALLOC), "float8_e4m3fn")
            acc = T.alloc_l0c((M, N), "float32")

            if normalize_to_k_major:
                T.copy(
                    a[0:actual_k, 0:M],
                    a_l1[0:M, 0:actual_k],
                    transpose=True,
                )
            elif major_mn:
                T.copy(a[0:actual_k, 0:M], a_l1[0:actual_k, 0:M])
            else:
                T.copy(a[0:M, 0:actual_k], a_l1[0:M, 0:actual_k])
            T.copy(b, b_l1)
            T.copy(a_l1, a_l0, scale=sfa_l1)
            T.copy(b_l1, b_l0, scale=sfb_l1)
            T.blockscaled_gemm(
                a_l0,
                b_l0,
                acc,
                transpose_A=major_mn and not normalize_to_k_major,
                transpose_B=True,
                clear_accum=True,
            )

    return kernel


@pytest.mark.parametrize(
    ("major_mn", "normalize_to_k_major", "actual_k", "offset", "repeats", "blocks", "gap"),
    [
        (False, False, 65, 6144, 1, 64, 0),
        (True, False, 65, 2080, 2, 63, 65),
        (True, True, 65, 6144, 1, 64, 0),
    ],
    ids=["major-k", "major-mn", "normalize-to-major-k"],
)
def test_mx_actual_k_copy_emits_deepgemm_tail_fill(
    major_mn,
    normalize_to_k_major,
    actual_k,
    offset,
    repeats,
    blocks,
    gap,
):
    source = tilelang.lower(
        make_mx_actual_k_copy(major_mn, actual_k, normalize_to_k_major),
        target="ascend",
    ).kernel_source
    fill = re.search(
        rf"asc_fill_l1\([^;]*\+ {offset}\), [^;]*, \{{\s*"
        rf"\.repeat = static_cast<uint64_t>\({repeats}\),\s*"
        rf"\.blk_num = static_cast<uint64_t>\({blocks}\),\s*"
        rf"\.dst_gap = static_cast<uint64_t>\({gap}\)\}}\);",
        source,
    )
    assert fill is not None, source
    assert source.count("asc_fill_l1(") == 1
    assert "asc_sync_pipe(PIPE_MTE1);" not in source


@pytest.mark.parametrize(
    ("major_mn", "normalize_to_k_major", "actual_k"),
    [(False, False, 33), (True, False, 64), (True, True, 33)],
)
def test_mx_actual_k_copy_skips_unneeded_tail_fill(major_mn, normalize_to_k_major, actual_k):
    source = tilelang.lower(
        make_mx_actual_k_copy(major_mn, actual_k, normalize_to_k_major),
        target="ascend",
    ).kernel_source
    assert "asc_fill_l1(" not in source


if __name__ == "__main__":
    tilelang.testing.main()
