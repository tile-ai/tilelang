import tilelang.testing

import example_gemm_sp


@tilelang.testing.requires_cuda
def test_example_gemm_sp():
    example_gemm_sp.main()


if __name__ == "__main__":
    tilelang.testing.main()
