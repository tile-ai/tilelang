# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import tilelang_example_sparse_tensorcore


@tilelang.testing.requires_cuda
def test_tilelang_example_sparse_tensorcore():
    tilelang_example_sparse_tensorcore.main()


if __name__ == "__main__":
    tilelang.testing.main()
