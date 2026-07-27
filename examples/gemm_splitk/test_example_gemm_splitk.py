import tilelang.testing

import example_tilelang_gemm_splitk
import example_tilelang_gemm_splitk_vectorize_atomicadd


def test_example_tilelang_gemm_splitk():
    example_tilelang_gemm_splitk.main()


def test_example_tilelang_gemm_splitk_vectorize_atomicadd():
    example_tilelang_gemm_splitk_vectorize_atomicadd.main()


if __name__ == "__main__":
    tilelang.testing.main()
