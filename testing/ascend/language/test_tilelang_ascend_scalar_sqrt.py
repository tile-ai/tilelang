"""Scalar sqrt lowering on Ascend."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def _scalar_rsqrt_kernel():
    """Build reciprocal sqrt from scalar division and sqrt."""

    @T.prim_func
    def main(value: T.float32, output: T.Tensor((1,), "float32")):
        """Apply reciprocal sqrt in ordinary AscendC scope."""

        with T.Kernel(1):
            output[0] = 1.0 / T.sqrt(value)

    return main


def _simt_sqrt_kernel():
    """Build a sqrt kernel inside a SimtVF region."""

    @T.prim_func
    def main(value: T.float32, output: T.Tensor((1,), "float32")):
        """Apply sqrt through the existing SIMT intrinsic path."""

        with T.Kernel(1), T.SimtVF(threads=1):
            output[0] = T.sqrt(value)

    return main


def _simt_rsqrt_kernel():
    """Build an rsqrt kernel inside a SimtVF region."""

    @T.prim_func
    def main(value: T.float32, output: T.Tensor((1,), "float32")):
        """Apply rsqrt through the existing SIMT intrinsic path."""

        with T.Kernel(1), T.SimtVF(threads=1):
            output[0] = T.rsqrt(value)

    return main


def test_scalar_rsqrt_codegen_uses_ascend_std_sqrt():
    """Lower scalar reciprocal sqrt to division by AscendC standard sqrt."""

    source = tilelang.lower(_scalar_rsqrt_kernel(), target="ascend").kernel_source

    assert "#include <utils/std/cmath.h>" in source
    assert "/ AscendC::Std::sqrt(" in source
    assert "rsqrtf" not in source


@pytest.mark.parametrize(
    "kernel_builder,intrinsic",
    [
        pytest.param(_simt_sqrt_kernel, "sqrtf", id="sqrt"),
        pytest.param(_simt_rsqrt_kernel, "rsqrtf", id="rsqrt"),
    ],
)
def test_simt_math_codegen_keeps_existing_intrinsics(kernel_builder, intrinsic):
    """Keep sqrt and rsqrt on their existing SIMT intrinsic path."""

    source = tilelang.lower(kernel_builder(), target="ascend").kernel_source

    assert "__simt_vf__" in source
    assert intrinsic in source
    assert "AscendC::Std::" not in source


def test_pto_scalar_rsqrt_codegen_keeps_existing_lowering():
    """Leave scalar sqrt unchanged for the PTO code generator."""

    source = tilelang.lower(_scalar_rsqrt_kernel(), target="pto").kernel_source

    assert "/ pto.sqrt(" in source
    assert "AscendC::Std::" not in source


@tilelang.testing.requires_ascend
def test_scalar_rsqrt_numerics():
    """Match scalar reciprocal sqrt against PyTorch."""

    kernel = tilelang.compile(_scalar_rsqrt_kernel(), target="ascend", out_idx=-1)
    value = 4.0
    actual = kernel(value).cpu()
    expected = torch.rsqrt(torch.tensor([value], dtype=torch.float32))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
