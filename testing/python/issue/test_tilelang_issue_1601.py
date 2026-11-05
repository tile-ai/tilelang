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
    source = kernel.get_kernel_source()
    assert "fp8_e4_t broadcast_var =" in source
    assert "tl::broadcast<fp8_e4_4_t>(broadcast_var)" in source
    data = torch.ones((8,), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    kernel(data)
    torch.cuda.synchronize()
    torch.testing.assert_close(data.view(torch.uint8), torch.zeros((8,), dtype=torch.uint8, device="cuda"), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
