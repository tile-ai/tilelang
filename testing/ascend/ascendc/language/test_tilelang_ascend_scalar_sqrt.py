"""Scalar sqrt and reciprocal sqrt on Ascend."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tvm import tirx
from testing.ascend._ir import nodes


def _math_kernel(math_op):
    @T.prim_func
    def main(value: T.float32, output: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            output[0] = math_op(value)

    return main


def _mixed_math_kernel(vf, math_op=T.rsqrt):
    @T.prim_func
    def main(value: T.float32, output: T.Tensor((3,), "float32")):
        with T.Kernel(1):
            output[0] = math_op(value)
            with vf():
                output[1] = math_op(value)
            output[2] = math_op(value)

    return main


@pytest.mark.parametrize("vf", [lambda: T.SimtVF(threads=1), T.SimdVF], ids=["simt", "simd"])
def test_rsqrt_frontend_composes_only_scalar_regions(vf):
    """The frontend emits division/sqrt only outside VF, without target binding."""
    func = _mixed_math_kernel(vf)
    stores = {int(store.indices[0]): store.value for store in nodes(func, tirx.BufferStore)}
    assert set(stores) == {0, 1, 2}
    for index in (0, 2):
        value = stores[index]
        assert isinstance(value, tirx.Div)
        assert value.dtype == value.a.dtype == value.b.dtype == "float32"
        assert isinstance(value.a, tirx.FloatImm) and float(value.a.value) == 1.0
        assert isinstance(value.b, tirx.Call) and value.b.op.name == "tirx.sqrt"
        assert value.b.args[0].same_as(func.params[0])
    assert isinstance(stores[1], tirx.Call) and stores[1].op.name == "tirx.rsqrt"


def test_scalar_rsqrt_codegen_selects_backend_late():
    """The same target-neutral frontend IR supports AscendC and PTO codegen."""
    func = _math_kernel(T.rsqrt)
    ascend_source = tilelang.lower(func, target="ascend").kernel_source
    pto_source = tilelang.lower(func, target="pto").kernel_source
    assert "/ AscendC::Std::sqrt(" in ascend_source
    assert "rsqrtf" not in ascend_source
    assert "/ pto.sqrt(" in pto_source
    assert "AscendC::Std::" not in pto_source


@pytest.mark.parametrize("math_op,intrinsic", [(T.sqrt, "sqrtf"), (T.rsqrt, "rsqrtf")], ids=["sqrt", "rsqrt"])
def test_mixed_scalar_simt_codegen_restores_execution_domain(math_op, intrinsic):
    """Codegen selects scalar math on both sides of a SIMT helper."""
    func = _mixed_math_kernel(lambda: T.SimtVF(threads=1), math_op)
    source = tilelang.lower(func, target="ascend").kernel_source
    assert source.count("#include <utils/std/cmath.h>") == 1
    assert source.count("AscendC::Std::sqrt(") == 2
    assert source.count(intrinsic + "(") == 1


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("math_op,reference", [(T.sqrt, torch.sqrt), (T.rsqrt, torch.rsqrt)], ids=["sqrt", "rsqrt"])
def test_scalar_math_numerics(math_op, reference):
    """Execute the public scalar math APIs on several positive inputs."""
    kernel = tilelang.compile(_math_kernel(math_op), target="ascend", out_idx=-1)
    for value in (0.25, 2.0, 4.0, 10.0, 1e-4, 1e4):
        actual = kernel(value).cpu()
        expected = reference(torch.tensor([value], dtype=torch.float32))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
