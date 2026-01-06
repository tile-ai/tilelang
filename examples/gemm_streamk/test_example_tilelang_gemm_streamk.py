import tilelang.testing

from example_tilelang_gemm_streamk import main


# not fully supported on sm90
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_tilelang_gemm_streamk():
    main()


if __name__ == "__main__":
    tilelang.testing.main()
