import tilelang.tools.bench
import example_tilelang_nsa_fwd
import example_tilelang_nsa_decode


def bench_example_tilelang_nsa_fwd():
    tilelang.tools.bench.process_func(example_tilelang_nsa_fwd.benchmark)


def bench_example_tilelang_nsa_fwd_decode():
    tilelang.tools.bench.process_func(example_tilelang_nsa_decode.benchmark)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
