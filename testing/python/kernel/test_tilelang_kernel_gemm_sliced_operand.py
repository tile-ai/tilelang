"""A shared operand that is a slice of a wider buffer must match the same GEMM
written with one buffer per slice.

Regression coverage for the WGMMA K-panel stride: a K-major operand whose
``block_K`` spans more than one swizzle atom is stored as (MN, (atom, panels)),
and stepping from one K panel to the next moves by the *buffer's* MN extent. The
offset formulas used to reconstruct that step from the operand's own extent,
which is smaller for a slice, so ``T.gemm(A_s, B_s[0:64, :], ...)`` silently read
the wrong panel for every ``ki`` past the first atom. ``block_K == 64`` (bf16 =
one 128B atom, single panel) never exercised the term and stayed correct.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

ACCUM = T.float32


def _matmul_sliced(M, N, K, block_M, block_N, block_K, dtype, split, operand, transposed, sliced):
    """Split one GEMM into two half-width ones over a shared operand.

    ``operand`` selects which operand is split ("A" splits M, "B" splits N),
    ``transposed`` gives that operand's ``transpose_`` flag (which decides whether
    the split axis is the buffer's leading or trailing one), and ``sliced`` picks
    the shape under test (one buffer consumed as two slices) versus the reference
    (one buffer per half).
    """
    trans_a = operand == "A" and transposed
    trans_b = operand == "B" and transposed
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    split_full = block_N if operand == "B" else block_M
    rest = split_full - split

    def a_tile(extent):
        return (block_K, extent) if trans_a else (extent, block_K)

    def b_tile(extent):
        return (extent, block_K) if trans_b else (block_K, extent)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            C0 = T.alloc_fragment((split, block_N) if operand == "A" else (block_M, split), ACCUM)
            C1 = T.alloc_fragment((rest, block_N) if operand == "A" else (block_M, rest), ACCUM)
            T.clear(C0)
            T.clear(C1)

            if operand == "B":
                A_s = T.alloc_shared(a_tile(block_M), dtype)
                if sliced:
                    B_s = T.alloc_shared(b_tile(block_N), dtype)
                else:
                    B_s0 = T.alloc_shared(b_tile(split), dtype)
                    B_s1 = T.alloc_shared(b_tile(rest), dtype)
            else:
                B_s = T.alloc_shared(b_tile(block_N), dtype)
                if sliced:
                    A_s = T.alloc_shared(a_tile(block_M), dtype)
                else:
                    A_s0 = T.alloc_shared(a_tile(split), dtype)
                    A_s1 = T.alloc_shared(a_tile(rest), dtype)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                if operand == "B":
                    if trans_a:
                        T.copy(A[ko * block_K, by * block_M], A_s)
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_s)
                    if sliced:
                        if trans_b:
                            T.copy(B[bx * block_N, ko * block_K], B_s)
                            T.gemm(A_s, B_s[0:split, :], C0, trans_a, trans_b)
                            T.gemm(A_s, B_s[split:block_N, :], C1, trans_a, trans_b)
                        else:
                            T.copy(B[ko * block_K, bx * block_N], B_s)
                            T.gemm(A_s, B_s[:, 0:split], C0, trans_a, trans_b)
                            T.gemm(A_s, B_s[:, split:block_N], C1, trans_a, trans_b)
                    else:
                        if trans_b:
                            T.copy(B[bx * block_N, ko * block_K], B_s0)
                            T.copy(B[bx * block_N + split, ko * block_K], B_s1)
                        else:
                            T.copy(B[ko * block_K, bx * block_N], B_s0)
                            T.copy(B[ko * block_K, bx * block_N + split], B_s1)
                        T.gemm(A_s, B_s0, C0, trans_a, trans_b)
                        T.gemm(A_s, B_s1, C1, trans_a, trans_b)
                else:
                    if trans_b:
                        T.copy(B[bx * block_N, ko * block_K], B_s)
                    else:
                        T.copy(B[ko * block_K, bx * block_N], B_s)
                    if sliced:
                        if trans_a:
                            T.copy(A[ko * block_K, by * block_M], A_s)
                            T.gemm(A_s[:, 0:split], B_s, C0, trans_a, trans_b)
                            T.gemm(A_s[:, split:block_M], B_s, C1, trans_a, trans_b)
                        else:
                            T.copy(A[by * block_M, ko * block_K], A_s)
                            T.gemm(A_s[0:split, :], B_s, C0, trans_a, trans_b)
                            T.gemm(A_s[split:block_M, :], B_s, C1, trans_a, trans_b)
                    else:
                        if trans_a:
                            T.copy(A[ko * block_K, by * block_M], A_s0)
                            T.copy(A[ko * block_K, by * block_M + split], A_s1)
                        else:
                            T.copy(A[by * block_M, ko * block_K], A_s0)
                            T.copy(A[by * block_M + split, ko * block_K], A_s1)
                        T.gemm(A_s0, B_s, C0, trans_a, trans_b)
                        T.gemm(A_s1, B_s, C1, trans_a, trans_b)

            if operand == "B":
                T.copy(C0, C[by * block_M, bx * block_N])
                T.copy(C1, C[by * block_M, bx * block_N + split])
            else:
                T.copy(C0, C[by * block_M, bx * block_N])
                T.copy(C1, C[by * block_M + split, bx * block_N])

    return main


def run_sliced_operand(M, N, K, block_M, block_N, block_K, dtype, split, operand, transposed):
    torch_dtype = getattr(torch, dtype)
    trans_a = operand == "A" and transposed
    trans_b = operand == "B" and transposed

    torch.manual_seed(0)
    a = torch.randn((K, M) if trans_a else (M, K), device="cuda", dtype=torch_dtype)
    b = torch.randn((N, K) if trans_b else (K, N), device="cuda", dtype=torch_dtype)
    ref = (a.T if trans_a else a).float() @ (b.T if trans_b else b).float()

    outputs = {}
    for sliced in (False, True):
        kernel = tilelang.compile(
            _matmul_sliced(M, N, K, block_M, block_N, block_K, dtype, split, operand, transposed, sliced),
            out_idx=[2],
            target="cuda",
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )
        outputs[sliced] = kernel(a, b).float()

    # The two shapes issue the same instructions over the same values, so they
    # agree bit-for-bit; the fp32 reference only bounds the accumulation error.
    torch.testing.assert_close(outputs[True], outputs[False], rtol=0, atol=0)
    torch.testing.assert_close(outputs[True], ref, rtol=1e-2, atol=1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("block_K", [64, 128, 256])
def test_gemm_sliced_shared_operand(operand, transposed, block_K):
    """block_K 128/256 put more than one swizzle atom on the K axis."""
    run_sliced_operand(
        M=256,
        N=256,
        K=512,
        block_M=128,
        block_N=128,
        block_K=block_K,
        dtype="bfloat16",
        split=64,
        operand=operand,
        transposed=transposed,
    )


if __name__ == "__main__":
    tilelang.testing.main()
