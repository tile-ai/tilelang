import tilelang.testing
import example_elementwise_add
import bench_example_elementwise


def test_example_elementwise_add():
    example_elementwise_add.main()



def test_bench_example_elementwise():
    bench_example_elementwise.main()
if __name__ == "__main__":
    tilelang.testing.main()