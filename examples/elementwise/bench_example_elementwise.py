import tilelang.tools.bench
import example_elementwise_add


def bench_example_elementwise_add():
    tilelang.tools.bench.process_func(example_elementwise_add.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
