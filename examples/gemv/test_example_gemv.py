# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_gemv


@tilelang.testing.requires_cuda
def test_example_gemv():
    example_gemv.main()


if __name__ == "__main__":
    tilelang.testing.main()
