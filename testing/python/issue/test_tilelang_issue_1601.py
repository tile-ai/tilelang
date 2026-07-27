import tilelang
import tilelang.testing
import tilelang.language as T


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
    assert "fp8_e4_t broadcast_var = fp8_e4_t(0x0p+0f/*0.000000e+00*/);" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
