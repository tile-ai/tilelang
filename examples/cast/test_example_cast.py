# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang.testing
import example_group_per_split_token_cast_to_fp8
import example_per_token_cast_to_fp8


def test_example_group_per_split_token_cast_to_fp8():
    example_group_per_split_token_cast_to_fp8.main()


def test_example_per_token_cast_to_fp8():
    example_per_token_cast_to_fp8.main()


if __name__ == "__main__":
    tilelang.testing.main()
