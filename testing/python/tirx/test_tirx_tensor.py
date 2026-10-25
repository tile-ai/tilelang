import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx

from tilelang._compat.tensor import TensorView, tensor_type


def test_tensor_view_preserves_native_identity():
    view = TensorView.create("tensor", [4, 8], "float32")
    assert view == TensorView(view.tensor)
    assert hash(view) == hash(TensorView(view.tensor))
    assert view != TensorView.create("tensor", [4, 8], "float32")
    assert view.dtype == "float32"
    assert view.scope() == "global"
    assert "tl.Buffer" not in ir.save_json(view.tensor)


def test_strides_and_offset_live_in_layout():
    ty = tensor_type([4, 8], "float32", strides=[20, 2], elem_offset=7)
    offset = sym.Analyzer().simplify(ty.layout.apply(2, 3)["m"])
    assert int(offset) == 53


def test_contiguous_layout():
    ty = tensor_type([4, 8], "float32")
    offset = sym.Analyzer().simplify(ty.layout.apply(2, 3)["m"])
    assert int(offset) == 19


def test_symbolic_shape_and_strides():
    size = ir.Var("size", "int32")
    ty = tensor_type([4, size], "float32", strides=[size + 1, 1])
    actual = sym.Analyzer().simplify(ty.layout.apply(2, 3)["m"])
    assert sym.Analyzer().can_prove_equal(actual, 2 * size + 5)


def test_geometry_has_one_owner():
    layout = tensor_type([8], "float32").layout
    with pytest.raises(ValueError, match="explicit layout"):
        tensor_type([8], "float32", layout=layout, strides=[2])
    with pytest.raises(ValueError, match="same rank"):
        tensor_type([4, 8], "float32", strides=[2])


def test_view_requires_tensor_variable():
    with pytest.raises(TypeError, match="TensorType"):
        TensorView(ir.Var("scalar", "int32"))


def test_allocate_is_an_explicit_bind():
    view = TensorView.create("scratch", [8], "float32", scope="local")
    binding = view.allocate()
    assert isinstance(binding, ir.Bind)
    assert binding.var.same_as(view.tensor)
    assert binding.value.op.name == "tirx.alloc_tensor"
    assert binding.value.ty == view.tensor.ty
    binding.value.validate()


def test_declaration_preserves_pointer_and_scope():
    pointer = ir.Var("pointer", ir.PtrType("float32", "global"))
    view = TensorView.create("view", [8], "float32")
    binding = view.declare(pointer)
    assert binding.value.op.name == "tirx.decl_tensor"
    assert binding.value.args[0].same_as(pointer)
    binding.value.validate()
    with pytest.raises(TypeError, match="Backing pointer"):
        view.declare(ir.Var("wrong", ir.PtrType("int32", "global")))
    with pytest.raises(TypeError, match="typed pointer"):
        view.declare(ir.Var("scalar", "int32"))


def test_load_store_order_and_rank():
    view = TensorView.create("tensor", [4, 8], "float32")
    load = view.load([1, 2])
    store = view.store([1, 2], 3.0)
    assert load.source.same_as(view.tensor)
    assert store.dest.same_as(view.tensor)
    assert [int(index) for index in store.indices] == [1, 2]
    assert float(store.value) == 3.0
    with pytest.raises(ValueError, match="Expected 2 indices"):
        view.load([1])


def test_region_and_pointer_retain_tensor():
    view = TensorView.create("tensor", [8], "float32")
    region = view.region([ir.Range(2, 6)])
    assert region.source.same_as(view.tensor)
    assert int(region.region[0].begin) == 2
    assert int(region.region[0].extent) == 4
    assert isinstance(view.pointer_at([2]).ty, ir.PtrType)


def test_native_allocation_cpu_end_to_end():
    source = TensorView.create("source", [16], "float32")
    destination = TensorView.create("destination", [16], "float32")
    scratch = TensorView.create("scratch", [16], "float32", scope="local")
    first_index = ir.Var("first_index", "int32")
    second_index = ir.Var("second_index", "int32")
    first = ir.For(first_index, 0, 16, 1, ir.ForKind.DEFAULT, scratch.store(first_index, source.load(first_index) + 2.0))
    second = ir.For(second_index, 0, 16, 1, ir.ForKind.DEFAULT, destination.store(second_index, scratch.load(second_index) * 3.0))
    function = tirx.Function(
        [source.tensor, destination.tensor],
        [scratch.allocate(), first, second],
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "tensor_bridge")
    executable = tirx.compile(function, target="c")
    input_array = np.arange(16, dtype="float32")
    actual = runtime.empty((16,), "float32")
    executable(runtime.tensor(input_array), actual)
    np.testing.assert_array_equal(actual.numpy(), (input_array + 2.0) * 3.0)


def test_native_strided_input_cpu_end_to_end():
    source = TensorView.create("source", [4, 8], "float32", strides=[16, 2], data_alignment=4)
    destination = TensorView.create("destination", [4, 8], "float32")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    body = ir.For(
        row,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        ir.For(column, 0, 8, 1, ir.ForKind.DEFAULT, destination.store([row, column], source.load([row, column]))),
    )
    function = tirx.Function([source.tensor, destination.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "strided_bridge"
    )
    executable = tirx.compile(function, target="c")
    input_array = np.arange(64, dtype="float32").reshape(4, 16)[:, ::2]
    actual = runtime.empty((4, 8), "float32")
    executable(tvm_ffi.from_dlpack(input_array), actual)
    np.testing.assert_array_equal(actual.numpy(), input_array)
