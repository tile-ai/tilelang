"""Dispatch predicates consume native MemoryScope without losing its spelling."""

import pytest
from tvm_tirx import ir, tirx

from tilelang.utils import language as utils
from tilelang.metal import utils as metal_utils


@pytest.mark.parametrize(
    "scope, expected",
    [
        ("global", {"global"}),
        ("shared", {"shared"}),
        ("shared.dyn", {"shared", "shared_dynamic"}),
        ("local", {"local"}),
        ("local.var", {"local_var"}),
        ("local.fragment", {"fragment"}),
        ("local.reducer", {"reducer"}),
        ("shared.tmem", {"tensor_memory"}),
        ("metal.simdgroup", {"metal_simdgroup"}),
        ("metal.cooperative_tensor", {"metal_cooperative_tensor"}),
    ],
)
@pytest.mark.parametrize("representation", ["tensor", "load", "region"])
def test_scope_dispatch(scope, expected, representation):
    tensor = ir.Var("tensor", tirx.TensorType((4,), "float32", scope=scope))
    operand = tensor
    if representation == "load":
        operand = ir.TensorLoad(tensor, [0])
    elif representation == "region":
        operand = ir.TensorRegion(tensor, [ir.Range(0, 4)], ir.TensorRegionType())
    for name in ("global", "shared", "shared_dynamic", "local", "local_var", "fragment", "reducer", "tensor_memory"):
        assert getattr(utils, "is_" + name)(operand) == (name in expected)
    for name in ("metal_simdgroup", "metal_cooperative_tensor"):
        assert getattr(metal_utils, "is_" + name)(operand) == (name in expected)
    assert utils.is_shared(operand, allow_dynamic=False) == (scope == "shared")
