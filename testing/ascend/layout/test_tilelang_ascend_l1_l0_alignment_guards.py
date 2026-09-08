"""Alignment-guard tests for Ascend L1->L0 copy inference.

Exercises the three static-divisibility ICHECKs in the Ascend copy layout
inference (src/ascend/op/copy.cc, dma_path 3/4):

  (1) blockscaled source L1 K axis must be divisible by 64      (k_align)
  (2) transposed source L1 MN axis must be divisible by C0      (fp8=32, bf16=16)
  (3) transposed source L1 K  axis must be divisible by 16

Transpose is triggered either explicitly (T.copy(..., transpose=True)) or
implicitly (a K-major L1 source feeding an MN-major L0 destination, which the
gemm operation selects from its transpose flags).

Positive cases only assert that compilation succeeds.  Negative cases assert
that compilation raises with the matching guard message.
"""

import pytest
import tilelang
import tilelang.language as T
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


def _l0_explicit_transpose_kernel(M, K, N, dtype="bfloat16"):
    """L1->L0 copy with an explicit transpose=True (k-on-row source)."""

    @T.prim_func
    def main(
        A: T.Buffer((K, M), dtype),
        B: T.Buffer((N, K), dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((K, M), dtype)
            b_l1 = T.alloc_l1((N, K), dtype)
            a_l0 = T.alloc_l0a((M, K), dtype)
            b_l0 = T.alloc_l0b((N, K), dtype)
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0, transpose=True)
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


def _compile(kernel):
    return tilelang.compile(kernel, out_idx=-1)


# ────────────────────────── positive cases ──────────────────────────
# Only assert that compilation succeeds (alignment guards do not fire).


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_bf16_implicit_transpose_aligned(trans_a, trans_b):
    # bf16: MN and K both 16-aligned (C0=16 for bf16).
    _compile(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "bfloat16"))


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_fp8_implicit_transpose_aligned(trans_a, trans_b):
    # fp8: MN must be 32-aligned (C0=32), K 16-aligned.  64 satisfies both.
    _compile(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "float8_e4m3fn"))


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
def test_fp32_implicit_transpose_aligned(trans_a, trans_b):
    # fp32: C0=8 but row fractal is 16.  The axis becoming L0 C0 must be
    # 8-aligned and the axis becoming L0 row16 must be 16-aligned.  K=N=M=64
    # satisfies both.
    _compile(_l0_gemm_kernel(64, 64, 64, trans_a, trans_b, "float32"))


def test_bf16_explicit_transpose_aligned():
    _compile(_l0_explicit_transpose_kernel(64, 64, 64, "bfloat16"))


def test_fp8_explicit_transpose_aligned():
    _compile(_l0_explicit_transpose_kernel(64, 64, 64, "float8_e4m3fn"))


def test_blockscaled_l0_k_aligned():
    # K=128 is divisible by 64.
    _compile(_blockscaled_l0_kernel(64, 128, 64))


# ────────────────────────── negative cases ──────────────────────────
# Assert compilation raises with the matching alignment-guard message.
#
# fp8 transpose packs two 16-row source fractals into one C0=32 group, so the
# source axis that becomes the L0 C0 axis must be an even number of 16-rows
# (i.e. divisible by 32).  K=48 is an odd multiple of 16 -> the guard fires.


def test_fp8_implicit_transpose_k_misaligned():
    # fp8 NN: B [K, N] k-major -> MN-major L0 (implicit transpose).  The K axis
    # becomes the L0 C0 axis; K=48 (odd x16) is not 32-aligned -> guard fires.
    with pytest.raises(Exception, match="allocation extent must be divisible by 32"):
        _compile(_l0_gemm_kernel(64, 48, 64, trans_a=False, trans_b=False, dtype="float8_e4m3fn"))


def test_fp8_explicit_transpose_k_misaligned():
    # Explicit transpose, fp8, A [K, M] with K=48 (odd x16, not 32-aligned).
    with pytest.raises(Exception, match="allocation extent must be divisible by 32"):
        _compile(_l0_explicit_transpose_kernel(64, 48, 64, "float8_e4m3fn"))


def test_bf16_transpose_k_misaligned():
    # bf16 transpose, K=24 (not 16-aligned) -> transpose alignment guard fires.
    with pytest.raises(Exception, match="allocation extent must be divisible by"):
        _compile(_l0_gemm_kernel(64, 24, 64, trans_a=True, trans_b=True, dtype="bfloat16"))


def test_fp32_transpose_row16_misaligned():
    # fp32 NN: B [K, N] k-major -> MN-major L0 (implicit transpose).  The N axis
    # becomes the L0 row16 axis and must be 16-aligned; N=24 is not -> guard
    # fires (fp32 C0=8, but the row-fractal direction still needs %16).
    with pytest.raises(Exception, match="allocation extent must be divisible by 16"):
        _compile(_l0_gemm_kernel(64, 64, 24, trans_a=False, trans_b=False, dtype="float32"))


def test_blockscaled_l0_k_misaligned():
    # Blockscaled K=96 (not 64-aligned) -> blockscaled K guard must fire.
    with pytest.raises(Exception, match="divisible by 64"):
        _compile(_blockscaled_l0_kernel(64, 96, 64))


if __name__ == "__main__":
    tilelang.testing.main()
