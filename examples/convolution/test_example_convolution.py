# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_convolution


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_convolution():
    example_convolution.main()


if __name__ == "__main__":
    tilelang.testing.main()
