import tilelang.testing
import example_elementwise_add


def test_example_elementwise_add():
    example_elementwise_add.main()


def test_example_elementwise_add_autotune():
    example_elementwise_add.main(use_autotune=True)


if __name__ == "__main__":
    tilelang.testing.main()
