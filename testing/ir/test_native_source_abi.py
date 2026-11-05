"""Source wrappers and runtimes use one native parameter/shape identity graph."""

import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang.jit.adapter.abi import SourceABIPlan
from tilelang.jit.adapter.utils import pythonic_expr


def tensor(name, shape, strides=None):
    layout = "default" if strides is None else tirx.TileLayout(S[shape:strides])
    return ir.Var(name, tirx.TensorType(shape, "float32", layout=layout))


def function(*parameters):
    return tirx.Function(list(parameters), [], ir.PrimType("void"))


@pytest.mark.parametrize("roundtrip", [False, True])
def test_direct_sources_and_abi_order_are_separate(roundtrip):
    n, m = ir.Var("n", "int64"), ir.Var("m", "int64")
    source = function(tensor("A", [2, 3], [n, 1]), tensor("B", [n]), tensor("C", [m]), n)
    if roundtrip:
        source = ir.load_json(ir.save_json(source))
    plan = SourceABIPlan(source)
    assert [var.name for var in plan.implicit_symbols] == ["m"]
    explicit_n = source.params[3]
    assert [value.tuple() for value in plan.sources[explicit_n]] == [(2, 3, -1, 1), (1, 0, 0, 1), (0, 1, 0, 1)]
    assert len(plan.parameters) == 4
    assert [param.kind for param in plan.parameters] == ["tensor", "tensor", "tensor", "scalar"]


def test_output_owner_does_not_reorder_implicit_arguments():
    n, m = ir.Var("n", "int64"), ir.Var("m", "int64")
    source = function(tensor("out", [n, m]), tensor("input_m", [m]), tensor("input_n", [n]))
    plan = SourceABIPlan(source)
    assert [value.name for value in plan.implicit_symbols] == ["n", "m"]
    primary = plan.primary_sources([0])
    assert primary[n].parameter_index == 2
    assert primary[m].parameter_index == 1
    assert [item.parameter_index for item in plan.sources[n]] == [0, 2]


def test_composite_shape_and_same_name_variables_retain_identity():
    n, other_n = ir.Var("n", "int64"), ir.Var("n", "int64")
    scalar = ir.Var("n", "int32")
    source = function(tensor("n", [n * 2 + other_n]), scalar)
    plan = SourceABIPlan(source)
    assert len(plan.implicit_symbols) == 2
    assert plan.implicit_symbols[0].same_as(n)
    assert plan.implicit_symbols[1].same_as(other_n)
    assert len({plan.name(value) for value in [source.params[0], scalar, n, other_n]}) == 4
    assert pythonic_expr(n + other_n, var_name_map=plan.names) == f"{plan.name(n)} + {plan.name(other_n)}"


def test_cache_metadata_is_checked_then_rebased_to_function_identity():
    n = ir.Var("n", "int64")
    source = function(tensor("A", [n]), tensor("B", [n * 2]))
    parameters = [KernelParam.from_var(parameter) for parameter in source.params]
    cached = load_kernel_params(dump_kernel_params(parameters))
    plan = SourceABIPlan(source)
    rebound = plan.validate_parameter_metadata(cached)
    assert rebound[0].shape[0].same_as(n)
    assert rebound[1].shape[0].a.same_as(n)
    cached[0].shape = [ir.Var("different", "int64")]
    with pytest.raises(ValueError, match="shape graph"):
        plan.validate_parameter_metadata(cached)


def test_tensor_pointer_scalar_and_rank_zero_are_distinct():
    source = function(tensor("value", []), ir.Var("value", ir.PtrType("uint8")), ir.Var("value", "uint32"))
    plan = SourceABIPlan(source)
    assert [parameter.kind for parameter in plan.parameters] == ["tensor", "pointer", "scalar"]
    assert list(plan.tensors) == [0]
    assert not plan.implicit_symbols
    assert len(set(plan.names.values())) == 3


def test_native_expression_precedence_and_cast_type():
    n, m = ir.Var("n", "int64"), ir.Var("m", "int64")
    assert pythonic_expr(n * (m + 3)) == "n * (m + 3)"
    assert pythonic_expr(n - (m - 1)) == "n - (m - 1)"
    assert pythonic_expr(ir.prim.Cast("uint32", n), expression_style="cxx") == "(uint32_t)n"
    assert pythonic_expr(ir.prim.floordiv(n, 8), floor_div_op="//") == "n // 8"
