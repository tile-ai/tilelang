import tilelang.tools.bench
import example_tilelang_nsa_fwd
import example_tilelang_nsa_decode


def bench_example_tilelang_nsa_fwd():
    tilelang.tools.bench.process_func(example_tilelang_nsa_fwd.main)


def bench_example_tilelang_nsa_fwd_decode():
    tilelang.tools.bench.process_func(example_tilelang_nsa_decode.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
