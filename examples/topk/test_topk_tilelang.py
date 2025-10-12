import tilelang.testing
import example_topk


@tilelang.testing.requires_cuda
def test_topk_tilelang():
    example_topk.main(args=[])


if __name__ == "__main__":
    tilelang.testing.main()
