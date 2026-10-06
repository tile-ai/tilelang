import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _row_offset_ldsm_gemm(M=16, N=16, K=16):
    """Shared -> fragment copy with a non-zero row offset on the shared side.

    ``ldmatrix`` reconstructs the shared address from the fragment layout's
    inverse alone, so it can only express a load that starts at row 0 of the
    shared tile. A copy out of ``As[M:2M, :]`` must therefore fall back to the
    normal (offset-aware) copy instead of being emitted as ``ptx_ldmatrix``.
    """

    @T.prim_func
    def main(
        A: T.Tensor((2 * M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=32):
            As = T.alloc_shared((2 * M, K), T.float16)
            Bs = T.alloc_shared((K, N), T.float16)
            A_frag = T.alloc_fragment((M, K), T.float16)
            Cf = T.alloc_fragment((M, N), T.float32)

            T.copy(A, As)
            T.copy(B, Bs)
            T.copy(As[M : 2 * M, :], A_frag)
            T.clear(Cf)
            T.gemm(A_frag, Bs, Cf)
            T.copy(Cf, C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_copy_shared_region_offset_not_lowered_to_ldmatrix():
    M, N, K = 16, 16, 16
    program = _row_offset_ldsm_gemm(M, N, K)
    kernel = tilelang.compile(program, target={"kind": "cuda", "arch": "sm_90"})

    torch.manual_seed(0)
    A = torch.randn(2 * M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    kernel(A, B, C)
    torch.cuda.synchronize()

    # The kernel must compute A[M:2M] @ B. Before the fix the row offset was
    # dropped and it silently computed A[0:M] @ B instead.
    torch.testing.assert_close(
        C,
        A[M : 2 * M].float() @ B.float(),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
