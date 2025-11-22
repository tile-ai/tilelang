import tilelang.tools.bench
import example_deepgemm_fp8_2xAcc


def bench_example_deepgemm_fp8_2xAcc():
    tilelang.tools.bench.process_func(example_deepgemm_fp8_2xAcc.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
