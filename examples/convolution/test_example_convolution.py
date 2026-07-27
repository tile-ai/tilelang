import tilelang.testing

import example_convolution
import example_convolution_autotune


# TODO(@cy): TMA with convolution must be fixed in future.
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_convolution():
    example_convolution.main([])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_convolution_autotune():
    example_convolution_autotune.main()


if __name__ == "__main__":
    tilelang.testing.main()
