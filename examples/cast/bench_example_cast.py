import tilelang.tools.bench
import example_group_per_split_token_cast_to_fp8
import example_per_token_cast_to_fp8


def bench_example_group_per_split_token_cast_to_fp8():
    tilelang.tools.bench.process_func(
        example_group_per_split_token_cast_to_fp8.benchmark,
        M=1024,
        N=1024,
        BG=2,
        blk_m=4,
        batch_sizes=[128, 896])


def bench_example_per_token_cast_to_fp8():
    tilelang.tools.bench.process_func(
        example_per_token_cast_to_fp8.benchmark, M=2048, N=512, blk_m=8)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
