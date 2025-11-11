import tilelang.testing
import example_dynamic
import bench_example_dynamic


def test_example_dynamic():
    example_dynamic.main(M=1024, N=1024, K=1024)


def test_bench_example_dynamic():
    bench_example_dynamic.main()


if __name__ == "__main__":
    tilelang.testing.main()
