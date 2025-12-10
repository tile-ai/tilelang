import tilelang.tools.bench
import example_linear_attn_bwd
import example_linear_attn_fwd


def bench_example_linear_attn_fwd():
    tilelang.tools.bench.process_func(example_linear_attn_fwd.benchmark)


def bench_example_linear_attn_bwd():
    tilelang.tools.bench.process_func(example_linear_attn_bwd.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
