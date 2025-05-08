# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing
import sys

import example_elementwise_add


def test_example_elementwise_add():
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        example_elementwise_add.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    tilelang.testing.main()
