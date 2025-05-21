# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_convolution


@tilelang.testing.requires_cuda
def test_example_convolution():
    example_convolution.main([])


if __name__ == "__main__":
    tilelang.testing.main()
