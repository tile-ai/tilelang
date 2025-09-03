import tilelang.testing
import example_topk


@tilelang.testing.requires_cuda
def test_topk_tilelang():
    example_topk.main()


if __name__ == "__main__":
    test_topk_tilelang()
