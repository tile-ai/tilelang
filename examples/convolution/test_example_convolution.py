import example_convolution
import example_convolution_autotune

import tilelang.testing


def test_example_convolution():
    example_convolution.main([])


def test_example_convolution_autotune():
    example_convolution_autotune.main()


if __name__ == "__main__":
    tilelang.testing.main()
