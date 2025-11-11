import tilelang.tools.bench
import example_tilelang_gemm_splitk
import example_tilelang_gemm_splitk_vectorize_atomicadd


def bench_example_tilelang_gemm_splitk():
    tilelang.tools.bench.process_func(example_tilelang_gemm_splitk.main)



def bench_example_tilelang_gemm_splitk_vectorize_atomicadd():
    tilelang.tools.bench.process_func(example_tilelang_gemm_splitk_vectorize_atomicadd.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
