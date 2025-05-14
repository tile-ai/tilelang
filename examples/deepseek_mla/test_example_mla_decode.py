# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_mla_decode


@tilelang.testing.requires_cuda
def test_example_mla_decode():
    example_mla_decode.main()


if __name__ == "__main__":
    tilelang.testing.main()