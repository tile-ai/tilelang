"""Zero-fill matching in the async copy injector.

A predicated staging store with an else-fill is a common masking idiom:

    for i, d in T.Parallel(M, N, prefer_async=True):
        S[i, d] = T.if_then_else(i < k_len, A[i, d], T.float32(-0.0))

`MatchZeroFillBufferLoad` recognises the else branch as a zero fill and lets the
copy take the async path. `-0.0` compares equal to `0.0`, so it used to match,
and the fill then wrote `+0.0` -- a different value, silently.
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

M, N = 64, 128
K_LEN = 50
NEGATIVE_ZERO_BITS = 0x80000000


def _make_masked_staging_kernel(prefer_async):
    @T.prim_func
    def main(
        A: T.Tensor([M, N], T.float32),
        Out: T.Tensor([M, N], T.float32),
        k_len: T.int32,
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared([M, N], T.float32)
            for i, d in T.Parallel(M, N, prefer_async=prefer_async):
                S[i, d] = T.if_then_else(i < k_len, A[i, d], T.float32(-0.0))
            for i, d in T.Parallel(M, N):
                Out[i, d] = S[i, d]

    return main


def _masked_row_bits(prefer_async):
    kernel = tilelang.compile(_make_masked_staging_kernel(prefer_async), out_idx=[1])
    A = torch.randn(M, N, dtype=torch.float32, device="cuda") + 1.0
    out = kernel(A, K_LEN)
    torch.cuda.synchronize()
    # Rows at and past k_len take the else-fill.
    return int(out.view(torch.int32)[K_LEN, 0].item()) & 0xFFFFFFFF


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_masked_staging_preserves_negative_zero_fill():
    """Both lowerings of the same kernel have to produce the same bytes."""
    assert _masked_row_bits(False) == NEGATIVE_ZERO_BITS

    bits = _masked_row_bits(True)
    assert bits == NEGATIVE_ZERO_BITS, f"async path stored {hex(bits)}, expected 0x80000000 (-0.0)"


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_masked_staging_preserves_positive_zero_fill():
    """Control: a `+0.0` fill still matches the zero-fill path."""

    @T.prim_func
    def main(
        A: T.Tensor([M, N], T.float32),
        Out: T.Tensor([M, N], T.float32),
        k_len: T.int32,
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared([M, N], T.float32)
            for i, d in T.Parallel(M, N, prefer_async=True):
                S[i, d] = T.if_then_else(i < k_len, A[i, d], T.float32(0.0))
            for i, d in T.Parallel(M, N):
                Out[i, d] = S[i, d]

    A = torch.randn(M, N, dtype=torch.float32, device="cuda") + 1.0
    bits = int(tilelang.compile(main, out_idx=[1])(A, K_LEN).view(torch.int32)[K_LEN, 0].item()) & 0xFFFFFFFF
    torch.cuda.synchronize()
    assert bits == 0


if __name__ == "__main__":
    tilelang.testing.main()
