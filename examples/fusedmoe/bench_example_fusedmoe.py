import tilelang.tools.bench
import example_fusedmoe_tilelang


def bench_example_fusedmoe_tilelang():
    tilelang.tools.bench.process_func(example_fusedmoe_tilelang.main)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
