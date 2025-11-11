import tilelang.tools.bench
import example_conv_analyze
import example_gemm_analyze


def bench_example_gemm_analyze():
    tilelang.tools.bench.process_func(example_gemm_analyze.main)



def bench_example_conv_analyze():
    tilelang.tools.bench.process_func(example_conv_analyze.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
