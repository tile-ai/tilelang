"""L1->L0 guards validate physical transpose groups and destination capacity.

Logical regions may end within a fractal. The emitted instruction must obey
the dtype's m_step/k_step constraints and fit the declared destination region.
"""

import re

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


# ─────────────────────────── kernel builders ───────────────────────────


def _l0_gemm_kernel(M, K, N, trans_a, trans_b, dtype="bfloat16"):
    """Single-tile L0 GEMM whose L1->L0 copies transpose implicitly when the
    K-major L1 source disagrees with the trans-derived MN-major L0 layout."""
    TKS = min(K, 64)
    SK = K // TKS

    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a_l1 = (K, M) if trans_a else (M, K)
    b_l1 = (N, K) if trans_b else (K, N)
    a_l0 = (TKS, M) if trans_a else (M, TKS)
    b_l0 = (N, TKS) if trans_b else (TKS, N)

    @T.prim_func
    def main(
        A: T.Buffer(a_shape, dtype),
        B: T.Buffer(b_shape, dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1) as bx:
            x_l0 = T.alloc_l0a(a_l0, dtype)
            w_l0 = T.alloc_l0b(b_l0, dtype)
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1(a_l1, dtype)
            w_l1 = T.alloc_l1(b_l1, dtype)
            temp = T.alloc_shared((M, N), "float32")
            for _ in T.Persistent([1], 1, bx):
                for kt in T.Pipelined(1, num_stages=1):
                    T.copy(A[:, :], x_l1)
                    T.copy(B[:, :], w_l1)
                    for sk in T.Pipelined(SK, num_stages=1):
                        if trans_a:
                            T.copy(x_l1[sk * TKS : (sk + 1) * TKS, :], x_l0)
                        else:
                            T.copy(x_l1[:, sk * TKS : (sk + 1) * TKS], x_l0)
                        if trans_b:
                            T.copy(w_l1[:, sk * TKS : (sk + 1) * TKS], w_l0)
                        else:
                            T.copy(w_l1[sk * TKS : (sk + 1) * TKS, :], w_l0)
                        T.gemm(x_l0, w_l0, res, transpose_A=trans_a, transpose_B=trans_b, clear_accum=(kt == 0 and sk == 0))
                T.copy(res, temp)
                T.copy(temp, C[:, :])

    return main


def _l0_explicit_transpose_kernel(M, K, N, dtype="bfloat16", mn_region=None, k_region=None, dst_k=None):
    """Explicit L1->L0 transpose, optionally restricting the source region."""
    tile_m = M if mn_region is None else mn_region
    tile_k = K if dst_k is None else dst_k
    copy_k = K if k_region is None else k_region

    @T.prim_func
    def main(
        A: T.Buffer((K, M), dtype),
        B: T.Buffer((N, tile_k), dtype),
        C: T.Buffer((tile_m, N), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((K, M), dtype)
            b_l1 = T.alloc_l1((N, tile_k), dtype)
            a_l0 = T.alloc_l0a((tile_m, tile_k), dtype)
            b_l0 = T.alloc_l0b((N, tile_k), dtype)
            acc = T.alloc_l0c((tile_m, N), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1[:copy_k, :tile_m], a_l0, transpose=True)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return main


def _blockscaled_l0_kernel(M, K, N):
    """Blockscaled (MX fp8) L0 GEMM, no transpose (NT)."""
    dtype = "float8_e4m3fn"
    sf_k = K // 64

    @T.prim_func
    def main(
        X: T.Buffer((M, K), dtype),
        W: T.Buffer((N, K), dtype),
        SFX: T.Buffer((M, sf_k), "uint16"),
        SFW: T.Buffer((N, sf_k), "uint16"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            xsf_l1 = T.alloc_l1((M, sf_k), "uint16")
            wsf_l1 = T.alloc_l1((N, sf_k), "uint16")
            x_l0 = T.alloc_l0a((M, K), dtype)
            w_l0 = T.alloc_l0b((N, K), dtype)
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.copy(SFX, xsf_l1)
            T.copy(SFW, wsf_l1)
            T.copy(x_l1, x_l0, scale=xsf_l1)
            T.copy(w_l1, w_l0, scale=wsf_l1)
            T.blockscaled_gemm(x_l0, w_l0, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return main


def _lower(kernel):
    return tilelang.lower(kernel, target="ascend")


# ────────────────────────── positive cases ──────────────────────────
# Lowering checks the descriptor without executing invalid kernels.


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_bf16_implicit_transpose_aligned(trans_a, trans_b):
    # bf16: MN and K both 16-aligned (C0=16 for bf16).
    _lower(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "bfloat16"))


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_fp8_implicit_transpose_aligned(trans_a, trans_b):
    # fp8 transpose needs an even m_step; complete64-wide tiles satisfy it.
    _lower(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "float8_e4m3fn"))


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_fp32_implicit_transpose_aligned(trans_a, trans_b):
    # fp32 transpose needs an even k_step; all source origins are aligned.
    _lower(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "float32"))


def test_bf16_explicit_transpose_aligned():
    _lower(_l0_explicit_transpose_kernel(64, 64, 64, "bfloat16"))


def test_fp8_explicit_transpose_aligned():
    _lower(_l0_explicit_transpose_kernel(64, 64, 64, "float8_e4m3fn"))


def test_blockscaled_l0_k_aligned():
    # K=128 is divisible by 64.
    _lower(_blockscaled_l0_kernel(64, 128, 64))


# ────────────────────────── negative cases ──────────────────────────
# Assert compilation raises with the matching alignment-guard message.
#
# fp8 transpose packs two 16-row source fractals into one C0=32 group, so the
# source axis that becomes the L0 C0 axis must be an even number of 16-rows
# (i.e. divisible by 32).  K=48 is an odd multiple of 16 -> the guard fires.


def test_fp8_implicit_transpose_k_misaligned():
    # fp8 NN: B [K, N] k-major -> MN-major L0 (implicit transpose).  The K axis
    # becomes the L0 C0 axis; K=48 (odd x16) is not 32-aligned -> guard fires.
    with pytest.raises(Exception, match="m_step to be divisible by 2"):
        _lower(_l0_gemm_kernel(64, 48, 64, trans_a=False, trans_b=False, dtype="float8_e4m3fn"))


def test_fp8_explicit_transpose_k_misaligned():
    # Explicit transpose, fp8, A [K, M] with K=48 (odd x16, not 32-aligned).
    with pytest.raises(Exception, match="m_step to be divisible by 2"):
        _lower(_l0_explicit_transpose_kernel(64, 48, 64, "float8_e4m3fn"))


def test_bf16_transpose_partial_fractal():
    # Both K24 and its destination occupy two bf16 K16 fractals.
    _lower(_l0_gemm_kernel(64, 24, 64, trans_a=True, trans_b=True, dtype="bfloat16"))


def test_fp32_transpose_row16_misaligned():
    # fp32 NN: source N24 produces three C0=8 groups, so k_step is odd.
    with pytest.raises(Exception, match="k_step to be divisible by 2"):
        _lower(_l0_gemm_kernel(64, 64, 24, trans_a=False, trans_b=False, dtype="float32"))


def test_blockscaled_l0_k_misaligned():
    # Blockscaled K=96 (not 64-aligned) -> blockscaled K guard must fire.
    with pytest.raises(Exception, match="divisible by 64"):
        _lower(_blockscaled_l0_kernel(64, 96, 64))


# ──────────────── region-level transpose guards (compact L0) ────────────────
# A padded allocation does not by itself make a partial copy descriptor legal.


def test_fp32_transpose_region_dst_k_padded():
    # Source K region 24 is rounded up to 32 by the transposed load; a padded
    # 32-wide L0A K allocation covers that rounding and is accepted.
    _lower(_l0_explicit_transpose_kernel(32, 32, 64, "float32", k_region=24))


def test_fp32_transpose_region_mn_misaligned():
    # Source MN24 in a padded32 allocation still emits k_step=3.
    with pytest.raises(Exception, match="k_step to be divisible by 2"):
        _lower(_l0_explicit_transpose_kernel(32, 32, 64, "float32", mn_region=24))


def test_fp32_transpose_region_dst_k_unpadded():
    # Source K region 24 -> the load writes 32 L0 K elements, which overflows
    # a 24-wide L0A K allocation.
    with pytest.raises(Exception, match="only covers 3x2"):
        _lower(_l0_explicit_transpose_kernel(32, 32, 64, "float32", k_region=24, dst_k=24))


def test_fp8_transpose_region_needs_two_source_row_blocks():
    # Padded allocation alone is insufficient: K16 emits m_step=1.
    with pytest.raises(Exception, match="m_step to be divisible by 2"):
        _lower(_l0_explicit_transpose_kernel(64, 64, 64, "float8_e4m3fn", k_region=16, dst_k=32))


def _guarded_transpose_kernel(parameter, active_misaligned=False):
    @T.prim_func
    def main(A: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            a_l0 = T.alloc_l0a((16, 16), "bfloat16")
            T.copy(A, a_l1)
            for i in T.serial(4):
                start = T.min(i * 16, 40)
                extent = T.min(16, 40 - start)
                origin = start + (8 if active_misaligned else 0)
                if parameter == "row_origin":
                    T.copy(a_l1[origin : origin + extent, :16], a_l0, transpose=True)
                else:
                    T.copy(a_l1[:16, origin : origin + extent], a_l0, transpose=True)

    return main


@pytest.mark.parametrize("parameter", ["row_origin", "c0_origin"])
def test_transpose_alignment_ignores_empty_iterations(parameter):
    source = _lower(_guarded_transpose_kernel(parameter)).kernel_source
    # A proof under has_data must not erase the runtime guard after its scope.
    assert re.search(r"if \([^\n]+\) \{\s*asc_copy_l12l0a_transpose\(", source), source


@pytest.mark.parametrize(
    "parameter,diagnostic",
    [("row_origin", "source row16 origin"), ("c0_origin", "source C0 origin")],
)
def test_transpose_alignment_still_rejects_nonempty_iterations(parameter, diagnostic):
    with pytest.raises(Exception, match=diagnostic):
        _lower(_guarded_transpose_kernel(parameter, active_misaligned=True))


@pytest.mark.parametrize("dtype,mn_region", [("bfloat16", 24), ("float32", 25)])
def test_transpose_partial_fractal_correctness(dtype, mn_region):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    kernel = tilelang.compile(_l0_explicit_transpose_kernel(32, 32, 64, dtype, mn_region=mn_region), out_idx=-1, target="ascend")
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (32, 32), generator=generator).to(getattr(torch, dtype)) / 8
    b = torch.randint(-4, 5, (64, 32), generator=generator).to(getattr(torch, dtype)) / 8
    expected = a[:, :mn_region].float().T @ b.float().T
    actual = kernel(a.npu(), b.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
