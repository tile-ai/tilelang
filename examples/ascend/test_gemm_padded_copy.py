import torch
import tilelang
import tilelang.language as T
import pytest


def gemm_padded_copy(M, K, N_pad):
    """GEMM where the L1 weight buffer (N_pad) is larger than the actual N,
    exercising the padded GM→CBuf dn2nz copy path whose dstNzC0Stride was
    previously computed from src dimensions (n_value) rather than dst
    dimensions (dst_n_value)."""

    N = N_pad - 16  # actual weight columns < buffer size

    @T.prim_func
    def main(
        X: T.Buffer((M, K), "bfloat16"),
        W: T.Buffer((K, N), "bfloat16"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1) as _:
            res = T.alloc_l0c((M, N_pad), "float32")
            x_l1 = T.alloc_l1((M, K), "bfloat16")
            w_l1 = T.alloc_l1((N_pad, K), "bfloat16")

            T.ascend_set_flag("MTE1_MTE2", 0)
            T.ascend_wait_flag("MTE1_MTE2", 0)

            T.copy(X[:, :], x_l1[:, :])
            T.copy(W[:, :], w_l1[:, :], transpose=True)

            T.ascend_set_flag("MTE2_MTE1", 0)
            T.ascend_wait_flag("MTE2_MTE1", 0)

            T.gemm(x_l1[:, :], w_l1[:, :], res, transpose_B=True, clear_accum=True)

            T.ascend_set_flag("M_FIX", 0)
            T.ascend_wait_flag("M_FIX", 0)

            T.copy(res[:, :N], C[:, :])

    return main


def ref_program(x, w):
    return x.float() @ w.float()


def _run_gemm_padded_copy(target):
    M, K = 256, 128
    N_pad = 112  # padded, N = N_pad - 16 = 96
    N = N_pad - 16

    kernel = tilelang.compile(
        gemm_padded_copy(M, K, N_pad),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


def test_gemm_padded_copy():
    _run_gemm_padded_copy("ascend")


@pytest.mark.pto
def test_gemm_padded_copy_pto():
    _run_gemm_padded_copy("pto")


if __name__ == "__main__":
    test_gemm_padded_copy()
    print("PASS: test_gemm_padded_copy")
    test_gemm_padded_copy_pto()
    print("PASS: test_gemm_padded_copy_pto")
