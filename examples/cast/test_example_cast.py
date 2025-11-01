import tilelang.testing
import example_group_per_split_token_cast_to_fp8
import example_per_token_cast_to_fp8


def test_example_group_per_split_token_cast_to_fp8():
    example_group_per_split_token_cast_to_fp8.main(M=8192, N=2048, BG=1, blk_m=4)


def test_example_per_token_cast_to_fp8():
    example_per_token_cast_to_fp8.main(M=2048, N=512, blk_m=8)


if __name__ == "__main__":
    tilelang.testing.main()
