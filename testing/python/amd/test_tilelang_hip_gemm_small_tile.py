"""T.gemm on tiles with few rows: clear errors, and the chain form that works."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch


@tilelang.jit(out_idx=[-1], target="hip")
def _single(M, N, K, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_l = T.alloc_fragment((M, N), T.float32)
            T.clear(C_l)
            T.copy(A[0, 0], A_s)
            T.copy(B[0, 0], B_s)
            T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[0, 0])

    return main


@tilelang.jit(out_idx=[-1], target="hip")
def _chain(M, P, D, threads, shared):
    @T.prim_func
    def main(
        Q: T.Tensor((M, D), T.float16),
        K: T.Tensor((P, D), T.float16),
        V: T.Tensor((P, D), T.float16),
        O: T.Tensor((M, D), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            Q_s = T.alloc_shared((M, D), T.float16)
            K_s = T.alloc_shared((P, D), T.float16)
            V_s = T.alloc_shared((P, D), T.float16)
            s = T.alloc_fragment((M, P), T.float32)
            p = T.alloc_shared((M, P), T.float16) if shared else T.alloc_fragment((M, P), T.float16)
            o = T.alloc_fragment((M, D), T.float32)
            T.copy(Q[0, 0], Q_s)
            T.copy(K[0, 0], K_s)
            T.copy(V[0, 0], V_s)
            T.clear(s)
            T.gemm(Q_s, K_s, s, transpose_B=True)
            for i, j in T.Parallel(M, P):
                p[i, j] = s[i, j]
            T.clear(o)
            T.gemm(p, V_s, o)
            T.copy(o, O[0, 0])

    return main


@tilelang.testing.requires_rocm
def test_too_many_wavefronts_reports_tile_limit():
    with pytest.raises(Exception, match=r"cannot split a 16x32 tile among 4 wavefronts"):
        _single(16, 32, 32, 256).get_kernel_source()


@tilelang.testing.requires_rocm
def test_fragment_chain_on_16_rows_explains_conflict():
    with pytest.raises(Exception, match=r"T\.alloc_shared memory"):
        _chain(16, 64, 128, 128, False).get_kernel_source()


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("threads", [64, 128, 256])
def test_chain_through_shared_memory(threads):
    M, P, D = 16, 64, 128
    kernel = _chain(M, P, D, threads, True)
    Q = torch.randn(M, D, dtype=torch.float16, device="cuda")
    K = torch.randn(P, D, dtype=torch.float16, device="cuda")
    V = torch.randn(P, D, dtype=torch.float16, device="cuda")
    out = kernel(Q, K, V)
    ref = (Q.float() @ K.float().T).half().float() @ V.float()
    assert (out.float() - ref).abs().max() <= 2e-2 * ref.abs().max()


if __name__ == "__main__":
    tilelang.testing.main()
