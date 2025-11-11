import tilelang.testing

from example_deepgemm_fp8_2xAcc import main
import bench_example_deepgemm_fp8_2xAcc


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_deepgemm_fp8_2xAcc():
    main()



def test_bench_example_deepgemm_fp8_2xAcc():
    bench_example_deepgemm_fp8_2xAcc.main()
if __name__ == "__main__":
    tilelang.testing.main()