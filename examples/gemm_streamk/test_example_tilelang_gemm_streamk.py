import tilelang.testing

from example_tilelang_gemm_streamk import main


@tilelang.testing.requires_cuda
def test_example_tilelang_gemm_streamk():
    main()


if __name__ == "__main__":
    tilelang.testing.main()
