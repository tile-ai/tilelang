# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing

import example_gqa_decode
import example_mha_inference


def test_example_example_gqa_decode():
    example_gqa_decode.main()


def test_example_example_mha_inference():
    example_mha_inference.main()


if __name__ == "__main__":
    tilelang.testing.main()