import tilelang.testing
import example_mla_decode


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mla_decode():
    example_mla_decode.main()


if __name__ == "__main__":
    tilelang.testing.main()
