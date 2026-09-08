"""TN matmul (A[K,M]^T @ B[K,N]) on Ascend, with auto-transpose L1->L0 copies.

Two variants:
  - major:          single-shot TN copy + gemm (K fits in one L0 tile)
  - subk_pipelined: L1 holds full K, L0A/L0B hold sub-K tiles filled inside a
                    T.Pipelined(num_stages=2) loop (double-buffer layout check)
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

M, N = 128, 128


@T.prim_func
def tn_gemm_kernel(
    A_gm: T.Buffer((64, M), "bfloat16"),
    B_gm: T.Buffer((64, N), "bfloat16"),
    C_gm: T.Buffer((M, N), "float32"),
):
    K = 64
    with T.Kernel(1):
        res = T.alloc_l0c((M, N), "float32")
        a_l1 = T.alloc_l1((K, M), "bfloat16")
        b_l1 = T.alloc_l1((K, N), "bfloat16")
        a_l0a = T.alloc_l0a((M, K), "bfloat16")
        b_l0b = T.alloc_l0b((N, K), "bfloat16")
        T.copy(A_gm, a_l1)
        T.copy(B_gm, b_l1)
        T.copy(a_l1, a_l0a, transpose=True)
        T.copy(b_l1, b_l0b, transpose=True)
        T.gemm(a_l0a, b_l0b, res, transpose_B=True, clear_accum=True)
        T.copy(res, C_gm)


@T.prim_func
def tn_subk_pipelined_kernel(
    A_gm: T.Buffer((128, M), "bfloat16"),
    B_gm: T.Buffer((128, N), "bfloat16"),
    C_gm: T.Buffer((M, N), "float32"),
):
    K = 128
    TILE_K_SUB = 64
    with T.Kernel(1):
        res = T.alloc_l0c((M, N), "float32")
        a_l1 = T.alloc_l1((K, M), "bfloat16")
        b_l1 = T.alloc_l1((K, N), "bfloat16")
        a_l0a = T.alloc_l0a((M, TILE_K_SUB), "bfloat16")
        b_l0b = T.alloc_l0b((N, TILE_K_SUB), "bfloat16")
        T.copy(A_gm, a_l1)
        T.copy(B_gm, b_l1)
        for sk in T.Pipelined(K // TILE_K_SUB, num_stages=2):
            T.copy(a_l1[sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB, :], a_l0a, transpose=True)
            T.copy(b_l1[sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB, :], b_l0b, transpose=True)
            T.gemm(a_l0a, b_l0b, res, transpose_B=True, clear_accum=(sk == 0))
        T.copy(res, C_gm)


def _run_tn_gemm(kernel_fn, K):
    torch.manual_seed(42)
    A = torch.randn(K, M, dtype=torch.bfloat16, device="npu")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
    expected = A.float().T @ B.float()

    kernel = tilelang.compile(kernel_fn, out_idx=-1)
    C = kernel(A, B)
    torch.npu.synchronize()

    rel = (C - expected).abs().max().item() / expected.abs().max().item()
    assert rel < 1e-2, f"rel={rel:.4e}"


def test_tn_gemm_major():
    _run_tn_gemm(tn_gemm_kernel, K=64)


def test_tn_subk_pipelined():
    _run_tn_gemm(tn_subk_pipelined_kernel, K=128)


if __name__ == "__main__":
    tilelang.testing.main()
