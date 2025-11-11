import tilelang.tools.bench
import example_topk

@tilelang.testing.requires_cuda
def bench_example_topk():
    tilelang.tools.bench.process_func(example_topk.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
