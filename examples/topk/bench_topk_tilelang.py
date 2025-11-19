import tilelang.tools.bench
import example_topk


def bench_example_topk():
    tilelang.tools.bench.process_func(example_topk.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
