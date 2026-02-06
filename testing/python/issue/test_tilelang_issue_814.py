import tilelang
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.jit
def _tmp_var_kernel(N, block_N, dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
            for i in T.Parallel(block_N):
                idx = bx * block_N + i
                tmp = T.max(A[idx], 1)
                B[idx] = tmp / 2
                A[idx] = tmp * 2

    return kernel


def run_tmp_var_test(N=1024, block_N=128):
    kernel = _tmp_var_kernel(N, block_N)

    a = torch.randn(N, device="cuda", dtype=torch.float)
    b = torch.empty(N, device="cuda", dtype=torch.float)

    a_ref = a.clone()

    kernel(a, b)

    # Reference computation
    tmp_ref = torch.maximum(a_ref, torch.tensor(1.0, dtype=torch.float, device="cuda"))
    b_ref = tmp_ref / 2
    a_ref = tmp_ref * 2

    # Validate correctness
    tilelang.testing.torch_assert_close(a, a_ref, rtol=1e-2, atol=1e-2)
    tilelang.testing.torch_assert_close(b, b_ref, rtol=1e-2, atol=1e-2)


def test_issue_814():
    """Test that temporary variables are correctly handled and not over-inlined"""
    run_tmp_var_test(N=1024, block_N=128)


if __name__ == "__main__":
    tilelang.testing.main()
