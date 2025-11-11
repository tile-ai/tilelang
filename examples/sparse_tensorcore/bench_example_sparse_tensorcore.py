import tilelang.tools.bench
import tilelang
import tilelang_example_sparse_tensorcore


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def bench_example_sparse_tensorcore():
    tilelang.tools.bench.process_func(tilelang_example_sparse_tensorcore.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
