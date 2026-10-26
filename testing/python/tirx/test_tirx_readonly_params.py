import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat.tensor import TensorView
from tilelang.transform import AnnotateReadOnlyParams

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def tensor(name):
    return TensorView.create(name, [16], "float32")


def annotate(parameters, body, *, previous=None):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([]))
    if previous is not None:
        function = function.with_attr("tl.readonly_param_indices", previous)
    return AnnotateReadOnlyParams()(ir.IRModule({"main": function}))["main"]


def readonly(function):
    return [int(index) for index in function.attrs["tl.readonly_param_indices"]]


def test_native_store_marks_only_destination():
    source, output = tensor("source"), tensor("output")
    scalar = ir.Var("scalar", "int32")
    result = annotate([source.tensor, scalar, output.tensor], output.store(0, source.load(0)))
    assert readonly(result) == [0]


def test_tile_copy_marks_only_its_declared_destination():
    source, output = tensor("source"), tensor("output")
    copy = ir.Evaluate(ir.Call("tl.tileop.copy", [source.tensor, output.tensor], ty="void"))
    assert readonly(annotate([source.tensor, output.tensor], copy)) == [0]


@pytest.mark.parametrize("pointer_kind", ["data", "element", "alias", "view"])
def test_opaque_call_through_pointer_and_tensor_alias(pointer_kind):
    source, output = tensor("source"), tensor("output")
    definitions = []
    if pointer_kind == "data":
        pointer = output.data
    elif pointer_kind == "element":
        pointer = output.pointer_at(1)
    elif pointer_kind == "alias":
        pointer = ir.Var("pointer", output.data.ty)
        definitions.append(ir.Bind(pointer, output.data))
    else:
        view = tensor("view")
        definitions.append(view.declare(output.data))
        pointer = view.pointer_at(1)
    call = tirx.call_extern("store_value", pointer, source.load(0), ty="void")
    result = annotate([source.tensor, output.tensor], [*definitions, ir.Evaluate(call)])
    assert readonly(result) == [0]


def test_store_through_native_decl_tensor_alias():
    source, output, view = tensor("source"), tensor("output"), tensor("view")
    result = annotate([source.tensor, output.tensor], [view.declare(output.data), view.store(0, source.load(0))])
    assert readonly(result) == [0]


def test_multiple_alias_bindings_preserve_pointer_origin():
    parameter = ir.Var("parameter", ir.PtrType("float32"))
    first = ir.Var("first", parameter.ty)
    second = ir.Var("second", parameter.ty)
    body = [ir.Bind(first, parameter), ir.Bind(second, first), ir.Evaluate(tirx.call_extern("write", second, ty="void"))]
    assert readonly(annotate([parameter], body)) == []


def test_pure_calls_do_not_mark_arguments_as_written():
    source, output = tensor("source"), tensor("output")
    call = tirx.call_pure_extern("inspect", source.data, ty="int32")
    result = annotate([source.tensor, output.tensor], [ir.Evaluate(call), output.store(0, source.load(0))])
    assert readonly(result) == [0]


def test_scalar_tensor_load_argument_is_not_a_write():
    source = tensor("source")
    result = annotate([source.tensor], ir.Evaluate(tirx.call_extern("consume", source.load(0), ty="void")))
    assert readonly(result) == [0]


def test_pointer_in_tuple_argument_remains_a_possible_write():
    source = tensor("source")
    argument = ir.Tuple([source.data, ir.const(1)])
    result = annotate([source.tensor], ir.Evaluate(tirx.call_extern("consume", argument, ty="void")))
    assert readonly(result) == []


def test_indirect_call_is_conservatively_effectful():
    pointer = ir.Var("pointer", ir.PtrType("float32"))
    callee = ir.Var("callee", ir.FuncType([pointer.ty], ir.TupleType([])))
    call = ir.Call(callee, [pointer])
    assert readonly(annotate([pointer, callee], ir.Evaluate(call))) == []


def test_temporary_tensor_writes_do_not_mark_inputs():
    source = tensor("source")
    temporary = TensorView.create("temporary", [16], "float32", scope="local")
    result = annotate([source.tensor], [temporary.allocate(), temporary.store(0, source.load(0))])
    assert readonly(result) == [0]


def test_stale_readonly_annotation_is_replaced_after_a_write():
    output = tensor("output")
    result = annotate([output.tensor], output.store(0, 1.0), previous=[ir.const(0)])
    assert readonly(result) == []


def test_native_scope_preserves_alias_analysis():
    source = tensor("source")
    alias = ir.Var("alias", source.data.ty)
    body = ir.RegionStmt("tl.scope", [], [], None, [ir.Bind(alias, source.data), ir.Evaluate(tirx.call_extern("write", alias, ty="void"))])
    assert readonly(annotate([source.tensor], body)) == []


def test_declaration_is_not_assumed_readonly():
    source = tensor("source")
    result = annotate([source.tensor], None)
    assert result.body is None
    assert "tl.readonly_param_indices" not in result.attrs


@pytest.mark.parametrize("mask", [0, 1, 2, 3])
@pytest.mark.parametrize("lower_pointer", [False, True])
@pytest.mark.parametrize("alias_depth", [0, 1, 2])
def test_typed_pointer_access_contract(mask, lower_pointer, alias_depth):
    source = tensor("source")
    pointer = tvm_ffi.get_global_func("tl.MakeAccessPtrFromBufferLoad")(source.load(0), mask)
    if lower_pointer:
        pointer = tvm_ffi.get_global_func("tl.LowerAccessPtrCall")(pointer)
    definitions = []
    for index in range(alias_depth):
        alias = ir.Var(f"pointer_{index}", pointer.ty)
        definitions.append(ir.Bind(alias, pointer))
        pointer = alias
    result = annotate([source.tensor], [*definitions, ir.Evaluate(tirx.call_extern("access", pointer, ty="void"))])
    assert readonly(result) == ([0] if mask in (0, 1) else [])


@pytest.mark.parametrize("lower_pointer", [False, True])
def test_alias_of_typed_access_pointer_retains_written_parameter_origin(lower_pointer):
    source = tensor("source")
    pointer = tvm_ffi.get_global_func("tl.MakeAccessPtrFromBufferLoad")(source.load(0), 3)
    if lower_pointer:
        pointer = tvm_ffi.get_global_func("tl.LowerAccessPtrCall")(pointer)
    alias = ir.Var("pointer", pointer.ty)
    result = annotate([source.tensor], [ir.Bind(alias, pointer), ir.Evaluate(tirx.call_extern("access", alias, ty="void"))])
    assert readonly(result) == []
