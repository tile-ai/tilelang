import tilelang
import tilelang.testing
import tilelang.language as T
import torch


def test_tmp_var(N, block_N, dtype="float"):

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
    func = test_tmp_var(N, block_N)
    jit_kernel = tilelang.compile(
        func,
        out_idx=[0, 1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True
        })

    a = torch.randn(N, device="cuda", dtype=torch.float)
    b = torch.empty(N, device="cuda", dtype=torch.float)

    a_ref = a.clone()

    jit_kernel(a, b)

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
