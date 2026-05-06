"""Regression tests for scalar-parameter input generation.

These tests cover `tilelang.utils.tensor.get_tensor_supply` for the
scalar (empty-shape) parameter case. Previously, calling the supplier
with a scalar `KernelParam` raised `ValueError`, which broke the
autotuner for any kernel signature that included a scalar value
parameter (e.g. `def kernel(A: T.Tensor(...), s: T.float32):`).

See: https://github.com/tile-ai/tilelang/issues/2081
"""

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.utils.tensor import TensorSupplyType, get_tensor_supply


# (dtype string, expected Python type for the supplied scalar)
_SCALAR_DTYPE_CASES = [
    ("float32", float),
    ("float16", float),
    ("bfloat16", float),
    ("float64", float),
    ("int32", int),
    ("int64", int),
    ("int8", int),
    ("uint8", int),
    ("bool", bool),
]


@pytest.mark.parametrize("dtype_str,expected_py_type", _SCALAR_DTYPE_CASES)
@pytest.mark.parametrize("supply_type", list(TensorSupplyType))
def test_scalar_param_returns_python_scalar(dtype_str, expected_py_type, supply_type):
    """A scalar `KernelParam` should yield a Python scalar of the right
    dtype family for every `TensorSupplyType`. This is the fallback
    that allows the autotuner to invoke kernels that take scalar
    value parameters; users can still supply explicit values via
    `supply_prog`. Regression for #2081.
    """
    param = KernelParam(dtype=tvm.DataType(dtype_str), shape=[])
    supply = get_tensor_supply(supply_type)

    value = supply(param)

    assert isinstance(value, expected_py_type), (
        f"Expected a {expected_py_type.__name__} for {dtype_str} scalar under {supply_type}, got {type(value).__name__} ({value!r})"
    )


def test_scalar_supply_does_not_require_cuda():
    """The scalar fast path must not depend on a CUDA device, so that
    autotuner input generation works on CPU-only hosts as well as on
    GPU machines."""
    param = KernelParam(dtype=tvm.DataType("float32"), shape=[])
    supply = get_tensor_supply(TensorSupplyType.Integer)
    # Should not raise, and should not touch CUDA at all.
    value = supply(param)
    assert isinstance(value, float)


if __name__ == "__main__":
    tilelang.testing.main()
