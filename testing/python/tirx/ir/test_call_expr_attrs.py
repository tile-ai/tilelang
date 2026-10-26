import pytest

from tvm_tirx import tirx
from tilelang._compat import ir
from tilelang._compat import structural


@pytest.fixture(scope="module", params=[False, True])
def annotated_operation(request):
    enabled = request.param
    name = f"test.call_expr_attrs.{str(enabled).lower()}"
    ir._ffi_api.RegisterOp(name, "Expression attribute traversal test")
    operation = ir.Op.get(name)
    operation.set_signature(args=[])
    operation.set_attr("TFixedReturnType", ir.TupleType([]))
    operation.set_attr("TCallEffectKind", 1)
    operation.set_attr("tl.attrs_may_contain_expr", enabled)
    return operation, enabled


def test_expression_attributes_are_opt_in(annotated_operation):
    operation, enabled = annotated_operation
    variable = ir.Var("extent", "int32")
    call = ir.Call(operation, [], attrs={"nested": {"extent": variable}})
    visited = []
    structural.structural_walk(call, (ir.Var, lambda value: visited.append(value)))
    assert bool(visited) == enabled
    if enabled:
        assert len(visited) == 1
        assert visited[0].same_as(variable)


def test_expression_attribute_rewriting_preserves_original(annotated_operation):
    operation, enabled = annotated_operation
    variable = ir.Var("extent", "int32")
    replacement = ir.Var("replacement", "int32")
    call = ir.Call(operation, [], attrs={"extent": variable})
    result = structural.structural_map(call, (ir.Var, lambda value: replacement if value.same_as(variable) else value))
    assert call.attrs["extent"].same_as(variable)
    assert result.attrs["extent"].same_as(replacement if enabled else variable)


def test_renew_def_remaps_expression_attributes():
    name = "test.call_expr_attrs.renew"
    ir._ffi_api.RegisterOp(name, "Definition renewal test")
    operation = ir.Op.get(name)
    operation.set_signature(args=[])
    operation.set_attr("TFixedReturnType", ir.TupleType([]))
    operation.set_attr("TCallEffectKind", 1)
    operation.set_attr("tl.attrs_may_contain_expr", True)
    variable = ir.Var("captured", "int32")
    function = tirx.Function(
        [],
        [
            ir.Bind(variable, ir.const(7)),
            ir.Evaluate(ir.Call(operation, [], attrs={"value": variable})),
        ],
    )
    result = ir.renew_def(function)
    renewed = result.body.seq[0].var
    assert not renewed.same_as(variable)
    assert result.body.seq[1].value.attrs["value"].same_as(renewed)
    assert ir.check_well_formed(result)
    ir.assert_structural_equal(function, result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
