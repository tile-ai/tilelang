import tilelang.testing
import example_topk
import bench_topk_tilelang


@tilelang.testing.requires_cuda
def test_topk_tilelang():
    example_topk.main(argv=[])


def test_bench_topk_tilelang():
    bench_topk_tilelang.main()


if __name__ == "__main__":
    tilelang.testing.main()
