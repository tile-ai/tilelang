import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def test_unroll_factor_codegen():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            for i in T.unroll(8, unroll_factor=4):
                A[i] = T.float32(1)
            for i in T.unroll(8):
                A[i + 8] = T.float32(2)

    source = lower(main, target="ascend").kernel_source

    assert "#pragma unroll 4\n" in source
    assert "#pragma unroll\n" in source


if __name__ == "__main__":
    tilelang.testing.main()
