import pytest
import torch
import tilelang.testing
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8_intrinsic
import example_tilelang_gemm_fp8


def requires_sm89():
    """FP8 tensor core MMA requires SM89 (Ada Lovelace) or higher."""
    major, minor = torch.cuda.get_device_capability()
    return pytest.mark.skipif(
        major < 9 and not (major == 8 and minor >= 9), reason="FP8 tensor core MMA requires SM89 or higher (Ada Lovelace/Hopper)"
    )


@requires_sm89()
def test_example_tilelang_gemm_fp8_2xAcc():
    example_tilelang_gemm_fp8_2xAcc.main()


@requires_sm89()
def test_example_tilelang_gemm_fp8_intrinsic():
    example_tilelang_gemm_fp8_intrinsic.main()


@requires_sm89()
def test_example_tilelang_gemm_fp8():
    example_tilelang_gemm_fp8.main()


if __name__ == "__main__":
    tilelang.testing.main()
