import example_blocksparse_gemm

import tilelang.testing


def test_example_blocksparse_gemm():
    example_blocksparse_gemm.main()


if __name__ == "__main__":
    tilelang.testing.main()
