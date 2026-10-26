import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir, sym

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def promote(expression):
    return tvm_ffi.get_global_func("tl.PromoteIndexToInt64")(expression)


def evaluate(expression, variable, value):
    replaced = stmt_functor.substitute(expression, {variable: ir.const(value, variable.ty)})
    return int(sym.Analyzer().simplify(replaced))


def test_index_arithmetic_is_widened_before_overflow():
    variable = ir.Var("value", "int32")
    expression = variable * variable + 1
    widened = promote(expression)
    assert widened.ty == ir.PrimType("int64")
    assert variable.ty == ir.PrimType("int32")
    assert evaluate(widened, variable, 50000) == 2500000001
    assert evaluate(widened, variable, -50000) == 2500000001


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32"])
def test_signed_variables_are_not_rebound(dtype):
    variable = ir.Var("value", dtype)
    widened = promote(variable)
    assert isinstance(widened, ir.prim.Cast)
    assert widened.ty == ir.PrimType("int64")
    assert widened.value.same_as(variable)


@pytest.mark.parametrize("dtype", ["uint8", "uint32", "uint64", "bool", "float32", "int64"])
def test_unaffected_variable_types_retain_identity(dtype):
    variable = ir.Var("value", dtype)
    assert promote(variable).same_as(variable)


def test_narrowing_cast_is_an_explicit_arithmetic_boundary():
    variable = ir.Var("value", "int32")
    narrowed = ir.prim.Cast("int8", variable + 1)
    widened = promote(narrowed + 2)
    assert widened.ty == ir.PrimType("int64")
    assert evaluate(widened, variable, 127) == -126
    retained = []
    stmt_functor.post_order_visit(
        widened, lambda node: retained.append(node) if isinstance(node, ir.prim.Cast) and node.ty == ir.PrimType("int8") else None
    )
    assert len(retained) == 1
    assert retained[0].value.ty == ir.PrimType("int32")


def test_let_binding_and_explicit_cast_preserve_lexical_identity():
    variable = ir.Var("input", "int32")
    bound = ir.Var("bound", "int32")
    expression = ir.prim.Let(bound, variable * variable, ir.prim.Cast("int8", bound) + bound)
    widened = promote(expression)
    assert widened.var.ty == ir.PrimType("int64")
    assert not widened.var.same_as(bound)
    assert evaluate(widened, variable, 50000) == 2500000000
    ir.assert_structural_equal(widened, ir.load_json(ir.save_json(widened)), map_free_vars=True)
    output = TensorView.create("output", [1], "int64")
    function = tirx.Function([variable, output.tensor], output.store(0, widened), ret_type=ir.TupleType([]))
    assert ir.check_well_formed(function)


def test_tensor_load_indices_are_promoted_without_changing_tensor_identity():
    tensor = TensorView.create("tensor", [128], "int32")
    index = ir.Var("index", "int32")
    original = tensor.load(index * 8)
    widened = promote(original)
    assert widened.source.same_as(tensor.tensor)
    assert widened.ty == original.ty
    assert widened.indices[0].ty == ir.PrimType("int64")


def test_widened_indices_execute_on_cpu(*, compile_tilelang):
    variable = ir.Var("input", "int32")
    output = TensorView.create("output", [1], "int64")
    widened = promote(variable * variable + 1)
    function = tirx.Function([variable, output.tensor], output.store(0, widened), ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "wide_index"
    )
    executable = compile_tilelang(function, target="c")
    actual = runtime.empty((1,), "int64")
    executable(50000, actual)
    np.testing.assert_array_equal(actual.numpy(), np.array([2500000001], dtype="int64"))
