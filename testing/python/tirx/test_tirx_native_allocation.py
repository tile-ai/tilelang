import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def transform(name, body, params):
    function = (
        tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "allocation").with_attr("target", Target("c"))
    )
    result = tvm_ffi.get_global_func(f"tl.transform.{name}")()(ir.IRModule({"main": function}))["main"]
    assert ir.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    for original, transformed in zip(params, result.params):
        if isinstance(original.ty, tirx.TensorType) and name == "FlattenBuffer":
            ir.assert_structural_equal(original.ty.shape, transformed.ty.shape)
            ir.assert_structural_equal(original.ty.dtype, transformed.ty.dtype)
            ir.assert_structural_equal(original.ty.scope, transformed.ty.scope)
        else:
            ir.assert_structural_equal(original.ty, transformed.ty)
    return result


def collect(function, node_type):
    nodes = []
    stmt_functor.post_order_visit(function, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


@pytest.mark.parametrize("shape", [[], [1], [1, 1]])
@pytest.mark.parametrize("use_hint", [False, True])
def test_allocation_owned_initializers_execute(shape, use_hint, *, compile_tilelang):
    local = TensorView.create("value", shape, "float32", scope="local")
    output = TensorView.create("output", [1], "float32")
    value = ir.Var("initial", "int32")
    if use_hint:
        statements = [local.allocate(), ir.Evaluate(ir.Call("tl.local_var_init_hint", [local.tensor, value]))]
    else:
        allocation = ir.Call("tirx.alloc_tensor", [], attrs={"tl.local_var_init": value}, ty_args=[local.tensor.ty])
        statements = [ir.Bind(local.tensor, allocation)]
    statements.append(output.store(0, local.load([0] * len(shape)) * 2))
    function = transform("LowerOpaqueBlock", statements, [output.tensor, value])
    assert not any(call.op.name == "tl.local_var_init_hint" for call in collect(function, ir.Call))
    executable = compile_tilelang(function, target="c")
    result = runtime.empty((1,), "float32")
    executable(result, 13)
    np.testing.assert_array_equal(result.numpy(), [26])


@pytest.mark.parametrize("shape,scope,diagnostic", [([2], "local", "single-element"), ([1], "global", "local storage")])
def test_invalid_initializer_storage_is_rejected(shape, scope, diagnostic):
    local = TensorView.create("value", shape, "int32", scope=scope)
    with pytest.raises(ValueError, match=diagnostic):
        transform("LowerOpaqueBlock", [local.allocate(), ir.Evaluate(ir.Call("tl.local_var_init_hint", [local.tensor, 7]))], [])


def test_unit_loop_preserves_lexical_allocation_and_step(*, compile_tilelang):
    local = TensorView.create("value", [1], "int32", scope="local")
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "int32")
    body = ir.For(index, 5, 8, 4, ir.ForKind.DEFAULT, [local.allocate(), local.store(0, index), output.store(0, local.load(0))])
    function = transform("LowerOpaqueBlock", body, [output.tensor])
    assert not collect(function, ir.For)
    assert len(collect(function, ir.RegionStmt)) == 1
    executable = compile_tilelang(function, target="c")
    result = runtime.empty((1,), "int32")
    executable(result)
    np.testing.assert_array_equal(result.numpy(), [5])


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_loop_control_is_not_removed(control):
    index = ir.Var("index", "int32")
    function = transform("LowerOpaqueBlock", ir.For(index, 0, 1, 1, ir.ForKind.DEFAULT, control()), [])
    assert len(collect(function, ir.For)) == 1
    assert len(collect(function, control)) == 1


def test_native_scope_metadata_survives():
    output = TensorView.create("output", [1], "int32")
    body = ir.RegionStmt("tl.scope", [], [], {"pipeline_stage": 3}, output.store(0, 5))
    function = transform("LowerOpaqueBlock", body, [output.tensor])
    scopes = collect(function, ir.RegionStmt)
    assert len(scopes) == 1
    assert scopes[0].attrs["pipeline_stage"] == 3


@pytest.mark.parametrize("dtype", ["int32", "float32", "bool"])
@pytest.mark.parametrize("offset", [0, 3])
def test_flatten_strided_aliases_keep_address_geometry(dtype, offset, *, compile_tilelang):
    storage = TensorView.create("storage", [32], dtype)
    view = TensorView.create("view", [2, 3], dtype, strides=[7, 2], elem_offset=offset)
    output = TensorView.create("output", [2, 3], dtype)
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    body = [
        view.declare(storage.data),
        ir.For(
            row,
            0,
            2,
            1,
            ir.ForKind.DEFAULT,
            ir.For(column, 0, 3, 1, ir.ForKind.DEFAULT, output.store([row, column], view.load([row, column]))),
        ),
    ]
    function = transform("FlattenBuffer", body, [storage.tensor, output.tensor])
    assert all(len(node.indices) == 1 for node in collect(function, ir.TensorLoad) + collect(function, ir.TensorStore))
    executable = compile_tilelang(function, target="c")
    values = (np.arange(32) % 3 == 0) if dtype == "bool" else np.arange(32).astype(dtype)
    actual = runtime.empty((2, 3), dtype)
    executable(runtime.tensor(values), actual)
    expected = values[offset + np.arange(2)[:, None] * 7 + np.arange(3)[None, :] * 2]
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_flatten_rank_zero_allocation(*, compile_tilelang):
    local = TensorView.create("scalar", [], "float32", scope="local")
    output = TensorView.create("output", [], "float32")
    function = transform("FlattenBuffer", [local.allocate(), local.store([], 4.5), output.store([], local.load([]))], [output.tensor])
    executable = compile_tilelang(function, target="c")
    actual = runtime.empty((), "float32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), 4.5)


def test_unrolled_allocation_initializers_follow_renewed_local_bindings(*, compile_tilelang):
    local = TensorView.create("local", [1], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index, value = ir.Var("index", "int32"), ir.Var("value", "int32")
    allocate = ir.Call("tirx.alloc_tensor", [], attrs={"tl.local_var_init": value}, ty_args=[local.tensor.ty])
    body = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.UNROLLED,
        [ir.Bind(value, index * 3), ir.Bind(local.tensor, allocate), output.store(index, local.load(0))],
        {"pragma_unroll_explicit": True},
    )
    unrolled = transform("UnrollLoop", body, [output.tensor])
    lowered = transform("LowerOpaqueBlock", unrolled.body, list(unrolled.params))
    executable = compile_tilelang(lowered, target="llvm")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(4) * 3)
