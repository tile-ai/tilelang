import tilelang.testing

import example_tilelang_gemm_splitk
import example_tilelang_gemm_splitk_vectorize_atomicadd
import bench_example_gemm_splitk


def test_example_tilelang_gemm_splitk():
    example_tilelang_gemm_splitk.main()


def test_example_tilelang_gemm_splitk_vectorize_atomicadd():
    example_tilelang_gemm_splitk_vectorize_atomicadd.main()



def test_bench_example_gemm_splitk():
    bench_example_gemm_splitk.main()
if __name__ == "__main__":
    tilelang.testing.main()