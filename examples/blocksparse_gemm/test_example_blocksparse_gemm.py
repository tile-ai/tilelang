# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing
import example_blocksparse_gemm


def test_example_blocksparse_gemm():
    example_blocksparse_gemm.main()


if __name__ == "__main__":
    tilelang.testing.main()
