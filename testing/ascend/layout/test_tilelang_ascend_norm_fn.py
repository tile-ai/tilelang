"""Simulate norm_fn fwd kernel's sub-K pattern:
x_l1[token_block, split_size] -> sub-K slice -> x_l0a[token_block, tile_k_sub]
fn_l1[mhc_mult3_pad16, split_size] -> sub-K slice -> fn_l0b[mhc_mult3_pad16, tile_k_sub]
gemm(x_l0a, fn_l0b, out_l0c, clear_accum=sk==0)
"""

import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

# Simplified norm_fn fwd dimensions
M = 64  # token_block
K = 128  # split_size
N = 32  # mhc_mult3_pad16
TILE_K_SUB = 64
NUM_SUB_K = K // TILE_K_SUB  # = 2


@T.prim_func
def norm_fn_fwd_pattern(
    X_gm: T.Buffer((M, K), "float32"),
    FN_gm: T.Buffer((N, K), "float32"),
    OUT_gm: T.Buffer((M, N), "float32"),
):
    with T.Kernel(1):
        x_l1 = T.alloc_l1((M, K), "float32")
        fn_l1 = T.alloc_l1((N, K), "float32")
        x_l0a = T.alloc_l0a((M, TILE_K_SUB), "float32")
        fn_l0b = T.alloc_l0b((N, TILE_K_SUB), "float32")
        out_l0c = T.alloc_l0c((M, N), "float32")

        T.copy(X_gm, x_l1)
        T.copy(FN_gm, fn_l1)

        T.set_hf32_mode("nearest_even")
        for sk in T.serial(NUM_SUB_K):
            # Sub-K slice from L1 to L0
            T.copy(x_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], x_l0a)
            T.copy(fn_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], fn_l0b)
            T.gemm(x_l0a, fn_l0b, out_l0c, transpose_B=True, clear_accum=(sk == 0))

        T.copy(out_l0c[:, :N], OUT_gm)


def test_norm_fn():
    torch.manual_seed(42)
    X = torch.randn(M, K, dtype=torch.float32, device="npu")
    FN = torch.randn(N, K, dtype=torch.float32, device="npu")
    expected = X @ FN.T

    kernel = tilelang.compile(norm_fn_fwd_pattern, out_idx=-1)
    OUT = kernel(X, FN)
    torch.npu.synchronize()

    rel = (OUT - expected).abs().max().item() / expected.abs().max().item()
    assert rel < 1e-2, f"rel={rel:.4e}"

    src = kernel.get_kernel_source()
    assert "mad(" in src, "missing cube mad instruction"
    assert "load_cbuf" in src, "missing L1->L0 load_cbuf"


if __name__ == "__main__":
    tilelang.testing.main()
