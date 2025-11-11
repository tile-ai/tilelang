import tilelang.tools.bench
import example_fusedmoe_tilelang


def bench_example_fusedmoe_tilelang():
    tilelang.tools.bench.process_func(example_fusedmoe_tilelang.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
