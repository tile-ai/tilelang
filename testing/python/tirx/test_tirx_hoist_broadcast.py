import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir.prim import Broadcast, Ramp

from tilelang.transform import HoistBroadcastValues


def transform(function):
    return HoistBroadcastValues()(ir.IRModule({"main": function}))["main"]


def test_hoist_broadcast_from_native_store():
    tensor = ir.Var("tensor", tirx.TensorType([4], "float32", layout="default"))
    store = ir.TensorStore(tensor, [Ramp(0, 1, 4)], Broadcast(ir.const(3.5, "float32"), 4))
    function = tirx.Function([tensor], store, ret_type=ir.TupleType([]))
    result = transform(function)
    definition, rewritten_store = result.body.seq
    assert isinstance(definition, ir.Bind)
    assert float(definition.value) == 3.5
    assert rewritten_store.value.value.same_as(definition.var)
    assert rewritten_store.dest.same_as(tensor)
    assert isinstance(function.body.seq[0].value.value, ir.FloatImm)


def test_hoist_broadcast_from_native_bind():
    vector = ir.Var("vector", "int32x4")
    function = tirx.Function([], ir.Bind(vector, Broadcast(ir.const(7), 4)), ret_type=ir.TupleType([]))
    result = transform(function)
    definition, vector_definition = result.body.seq
    assert definition.value.value == 7
    assert vector_definition.var.same_as(vector)
    assert vector_definition.value.value.same_as(definition.var)


def test_hoisted_bind_remains_inside_loop():
    tensor = ir.Var("tensor", tirx.TensorType([16], "float32", layout="default"))
    index = ir.Var("index", "int32")
    store = ir.TensorStore(tensor, [Ramp(index * 4, 1, 4)], Broadcast(ir.const(3.5, "float32"), 4))
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, store)
    result = transform(tirx.Function([tensor], loop, ret_type=ir.TupleType([])))
    assert len(result.body.seq) == 1
    assert isinstance(result.body.seq[0], ir.For)
    assert isinstance(result.body.seq[0].body.seq[0], ir.Bind)


def test_broadcast_variable_and_declarations_are_unchanged():
    scalar = ir.Var("scalar", "int32")
    vector = ir.Var("vector", "int32x4")
    function = tirx.Function([scalar], ir.Bind(vector, Broadcast(scalar, 4)), ret_type=ir.TupleType([]))
    assert tvm_ffi.structural_equal(function, transform(function))
    declaration = tirx.Function([scalar], None, ret_type=ir.TupleType([]))
    assert tvm_ffi.structural_equal(declaration, transform(declaration))
