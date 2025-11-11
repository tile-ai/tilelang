import tilelang.testing

from example_tilelang_gemm_streamk import main
import bench_example_tilelang_gemm_splitk


# not fully supported on sm90
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_tilelang_gemm_streamk():
    main()


def test_bench_example_tilelang_gemm_splitk():
    bench_example_tilelang_gemm_splitk.main()


if __name__ == "__main__":
    tilelang.testing.main()
