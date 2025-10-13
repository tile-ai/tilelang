import tilelang.testing
import example_topk


@tilelang.testing.requires_cuda
def test_topk_tilelang():
    example_topk.main(argv=[])


if __name__ == "__main__":
    tilelang.testing.main()
