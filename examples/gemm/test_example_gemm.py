import example_gemm
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm_schedule

import tilelang.testing


def test_example_gemm_autotune():
    # enable roller for fast tuning
    example_gemm_autotune.main(with_roller=True)


def test_example_gemm_intrinsics():
    example_gemm_intrinsics.main()


def test_example_gemm_schedule():
    example_gemm_schedule.main()


def test_example_gemm():
    example_gemm.main()


if __name__ == "__main__":
    tilelang.testing.main()
