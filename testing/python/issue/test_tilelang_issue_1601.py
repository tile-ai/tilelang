import tilelang
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.testing.requires_cuda
def test_issue_1601():
    @tilelang.jit
    def qwq():
        @T.prim_func
        def main(
            A: T.Tensor((8,), T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=32):
                for i in T.vectorized(8):
                    A[i] = 0

        return main

    kernel = qwq()
    assert "__nv_fp8_e4m3" in kernel.get_kernel_source()
    values = torch.full((8,), 3.0, device="cuda").to(torch.float8_e4m3fn)
    kernel(values)
    torch.testing.assert_close(values.float(), torch.zeros(8, device="cuda"), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
