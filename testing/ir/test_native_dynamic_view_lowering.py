"""LowerTileOp keeps each typed view use attached to its rewritten definition."""

import ctypes

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang import transform
from tilelang.testing.packed import compile_packed_function
from tilelang.language._native import buffer


def _tensor(name, shape, scope="global"):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope))


def _input(kind, scope):
    storage = _tensor("storage", (32,))
    output = _tensor("output", (3,))
    metadata = _tensor("metadata", (1,), scope)
    load = ir.TensorLoad(metadata, [0])
    if kind == "stride":
        ty = buffer((8,), "int32", strides=(load,)).as_type()
        indices = [2]
        positions = [2, 4, 2]
    elif kind == "offset":
        ty = buffer((8,), "int32", elem_offset=load).as_type()
        indices = [2]
        positions = [3, 4, 3]
    else:
        ty = buffer((2, load), "int32").as_type()
        indices = [1, 0]
        positions = [1, 2, 1]
    view = ir.Var("view", ty)
    params = [storage, output]
    body = []
    if scope == "global":
        params.append(metadata)
    else:
        body.append(ir.Bind(metadata, ir.Call("tirx.alloc_tensor", [], ty_args=[metadata.ty])))
    body += [
        ir.TensorStore(metadata, [0], 1),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [tirx.tensor_data_ptr(storage)], ty_args=[view.ty])),
    ]
    return params, body, storage, output, metadata, view, indices, positions


def _lower(function):
    target = tvm_tirx.target.Target("c")
    function = function.with_attr({"target": target, "global_symbol": "main"})
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    result = transform.LowerTileOp()(ir.IRModule({"main": function}))
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    # A second invocation must retain both the definition and its uses.
    ir.assert_structural_equal(result, transform.LowerTileOp()(result))
    return result, target


def _compile(function):
    module, target = _lower(function)
    module = tirx.transform.LowerTensorLayouts()(module)
    module = tirx.transform.FlattenTensor()(module)
    module = transform.LowerAccessPtr()(module)
    module = transform.FinalizeAnnotations("c")(module)
    assert tirx.analysis.check_well_formed(module)
    return compile_packed_function(module, tilelang_owned=False, target=target)


@pytest.mark.parametrize("kind", ["stride", "offset", "shape"])
@pytest.mark.parametrize("scope", ["global", "local.fragment"])
@pytest.mark.parametrize("access", ["load", "store"])
def test_dynamic_view_metadata_is_read_at_each_access(kind, scope, access):
    params, body, storage, output, metadata, view, indices, positions = _input(kind, scope)
    for index, value in enumerate((1, 2, 1)):
        body.append(ir.TensorStore(metadata, [0], value))
        if access == "store":
            body.append(ir.TensorStore(view, indices, 101 + index))
        body.append(ir.TensorStore(output, [index], ir.TensorLoad(view, indices)))
    executable = _compile(tirx.Function(params, body))
    values = np.arange(32, dtype="int32") + 10
    tensors = [tvm_tirx.runtime.tensor(values), tvm_tirx.runtime.tensor(np.full(3, -99, dtype="int32"))]
    if scope == "global":
        tensors.append(tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32")))
    executable(*tensors)
    expected = values.copy()
    if access == "load":
        expected_output = values[positions]
    else:
        expected_output = np.arange(101, 104, dtype="int32")
        for index, position in enumerate(positions):
            expected[position] = 101 + index
    np.testing.assert_array_equal(tensors[0].numpy(), expected)
    np.testing.assert_array_equal(tensors[1].numpy(), expected_output)


@pytest.mark.parametrize("kind", ["stride", "offset", "shape"])
@pytest.mark.parametrize("scope", ["global", "local.fragment"])
def test_dynamic_view_address_uses_the_current_metadata(kind, scope):
    params, body, storage, output, metadata, view, indices, positions = _input(kind, scope)
    body.append(ir.TensorStore(metadata, [0], 2))
    address = ir.Call("tl.access_ptr", [ir.TensorLoad(view, indices), 1, 1])
    body.append(ir.Return(address))
    executable = _compile(tirx.Function(params, body, address.ty))
    tensors = [
        tvm_tirx.runtime.tensor(np.arange(32, dtype="int32")),
        tvm_tirx.runtime.tensor(np.zeros(3, dtype="int32")),
    ]
    if scope == "global":
        tensors.append(tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32")))
    result = executable(*tensors)
    value = result.value if isinstance(result, ctypes.c_void_p) else int(result)
    assert value == tensors[0].data_ptr() + positions[1] * 4


@pytest.mark.parametrize("kind", ["stride", "offset", "shape"])
@pytest.mark.parametrize("scope", ["global", "local.fragment"])
def test_dynamic_tensor_regions_refer_to_the_same_view_definition(kind, scope):
    params, body, storage, output, metadata, view, indices, positions = _input(kind, scope)
    body.append(ir.Evaluate(ir.TensorRegion(view, [ir.Range.from_min_extent(index, 1) for index in indices], ir.TensorRegionType())))
    module, _ = _lower(tirx.Function(params, body))
    definitions = []
    regions = []
    tvm_ffi.structural_walk(module, (ir.Bind, lambda node: definitions.append(node.var)))
    tvm_ffi.structural_walk(module, (ir.TensorRegion, regions.append))
    assert len(regions) == 1
    assert any(regions[0].source.same_as(var) for var in definitions)
