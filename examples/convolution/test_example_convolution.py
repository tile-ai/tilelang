import tilelang.testing

import example_convolution
import example_convolution_autotune


@tilelang.testing.requires_cuda
def test_example_convolution():
    example_convolution.main([])


@tilelang.testing.requires_cuda
def test_example_convolution_autotune():
    example_convolution_autotune.main()


if __name__ == "__main__":
    tilelang.testing.main()
