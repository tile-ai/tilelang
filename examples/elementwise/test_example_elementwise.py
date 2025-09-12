import tilelang.testing
import example_elementwise_add
import example_elementwise_add_tma_1d


def test_example_elementwise_add():
    example_elementwise_add.main()


def test_example_elementwise_add_tma_1d():
    example_elementwise_add_tma_1d.main()


if __name__ == "__main__":
    tilelang.testing.main()
