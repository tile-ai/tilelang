"""Read-only parameter annotations must follow physical storage and calls."""

import pytest
from tvm_tirx import ir, tirx

import tilelang.transform as tr


def make_tensors():
    return [ir.Var(name, tirx.TensorType([4], "int32", data_alignment=4)) for name in ["a", "b", "c"]]


def readonly(params, body, noalias=True, exceptions=()):
    function = tirx.Function(params, body, ir.NoneType()).with_attr("tirx.noalias", noalias)
    if exceptions:
        function = function.with_attr("tl.non_restrict_params", list(exceptions))
    result = tr.AnnotateReadOnlyParams()(ir.IRModule({"main": function}))["main"]
    return list(map(int, result.attrs.get("tl.readonly_param_indices", [])))


@pytest.mark.parametrize("noalias,expected", [(True, [0, 2]), (False, [])])
def test_readonly_accounts_for_function_alias_contract(noalias, expected):
    a, b, c = make_tensors()
    assert readonly([a, b, c], [ir.TensorStore(b, [0], ir.TensorLoad(a, [0]))], noalias) == expected


def test_nonrestrict_parameters_share_writes_but_unrelated_parameter_does_not():
    a, b, c = make_tensors()
    assert readonly([a, b, c], [ir.TensorStore(b, [0], ir.TensorLoad(a, [0]))], exceptions=[a, b]) == [2]


def test_store_through_a_renamed_view_marks_original_parameter():
    a, b, c = make_tensors()
    view = ir.Var("view", a.ty)
    bind = ir.Bind(view, ir.Call("tirx.decl_tensor", [a.data], ty_args=[view.ty]))
    assert readonly([a, b, c], [bind, ir.TensorStore(view, [0], ir.IntImm("int32", 3))]) == [1, 2]


def test_opaque_no_pointer_call_may_write_previously_escaped_storage():
    a, b, c = make_tensors()
    call = tirx.call_extern("mutate_saved_pointer", ty="void")
    assert readonly([a, b, c], [ir.Evaluate(call)]) == []


def test_explicit_access_read_mask_does_not_imply_a_store():
    a, b, c = make_tensors()
    access = ir.Call("tl.access_ptr", [ir.TensorLoad(a, [0]), ir.IntImm("int32", 1), ir.IntImm("int32", 1)])
    access.validate()
    assert readonly([a, b, c], [ir.Evaluate(access)]) == [0, 1, 2]


def _metal_fragment(name):
    return ir.Var(name, tirx.TensorType((8, 8), "float32", scope="metal.cooperative_tensor"))


def _metal_transfer(kind, fragment, pointer):
    call = ir.Call(
        f"tl.cooperative_tensor_{kind}",
        [fragment, 0, pointer, 8, 8, 8, False, 8, 8, 8, 0],
        ty="void",
    )
    call.validate()
    return ir.Evaluate(call)


@pytest.mark.parametrize("kind,expected", [("load", [0, 2]), ("store", [1, 2])])
@pytest.mark.parametrize("noalias", [True, False])
def test_metal_transfer_marks_only_its_destination(kind, expected, noalias):
    data = ir.Var("data", ir.PtrType("float32"))
    fragment = _metal_fragment("fragment")
    other = ir.Var("other", ir.PtrType("float32"))
    body = [_metal_transfer(kind, fragment, data)]
    assert readonly([data, fragment, other], body, noalias) == (expected if noalias else [])


def test_metal_transfer_through_view_preserves_the_source_parameter_constness():
    source = ir.Var("source", tirx.TensorType((64,), "float32"))
    output = ir.Var("output", tirx.TensorType((64,), "float32"))
    fragment = _metal_fragment("fragment")
    view = ir.Var("output_view", output.ty)
    body = [
        ir.Bind(fragment, ir.Call("tirx.alloc_tensor", [], ty_args=[fragment.ty])),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [output.data], ty_args=[view.ty])),
        _metal_transfer("load", fragment, source.data),
        _metal_transfer("store", fragment, view.data),
    ]
    assert readonly([source, output], body) == [0]
    assert readonly([source, output], body, exceptions=[source, output]) == []


def test_metal_fill_and_mma_keep_read_operands_readonly():
    output, lhs, rhs, accum = [_metal_fragment(name) for name in ("output", "lhs", "rhs", "accum")]
    fill = ir.Call("tl.cooperative_tensor_fill", [output, 0, ir.const(0.0, "float32"), 8, 8], ty="void")
    mma = ir.Call(
        "tl.cooperative_tensor_multiply_accumulate",
        [output, 0, lhs, 0, rhs, 0, accum, 0, 8, 8, 8, False, False],
        ty="void",
    )
    fill.validate()
    mma.validate()
    assert readonly([output, lhs, rhs, accum], [ir.Evaluate(fill), ir.Evaluate(mma)]) == [1, 2, 3]


def test_metal_known_effect_does_not_hide_an_effectful_pointer_producer():
    source = ir.Var("source", ir.PtrType("float32"))
    fragment = _metal_fragment("fragment")
    escaped = tirx.call_extern("mutate_and_get_pointer", ty=ir.PtrType("float32"))
    body = [_metal_transfer("load", fragment, escaped)]
    assert readonly([source, fragment], body) == []
