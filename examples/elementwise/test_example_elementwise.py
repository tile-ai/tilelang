# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_elementwise_add


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_elementwise_add():
    example_elementwise_add.main()


if __name__ == "__main__":
    tilelang.testing.main()
