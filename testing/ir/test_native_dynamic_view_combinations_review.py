"""Independent review of physical layout and native definition remap together."""

import numpy as np
import pytest
import tvm_ffi
from test_native_dynamic_view_lowering import _compile, _lower
from tilelang.layout import Layout

import tvm_tirx
from tvm_tirx import ir, tirx


def _tensor(name, shape, scope="global", layout=None):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope, layout=layout))


def _allocation(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def _annotation(fields):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def _input(mapping, metadata_scope):
    metadata = _tensor("metadata", [1], metadata_scope)
    scope = "global" if mapping == "layout_map" else "local.fragment"
    storage = _tensor("storage", [32], scope)
    output = _tensor("output", [3])
    dynamic = ir.TensorLoad(metadata, [0])
    view = _tensor("view", [8], scope, tirx.TileLayout(tirx.layout.S[(8,) : (1,)] + dynamic))
    params, body = [output], []
    if scope == "global":
        params.append(storage)
    else:
        body.append(_allocation(storage))
        index = ir.Var("i", "int32")
        body.append(ir.For(index, 0, 32, 1, ir.ForKind.DEFAULT, [ir.TensorStore(storage, [index], index + 10)]))
    if metadata_scope == "global":
        params.append(metadata)
    else:
        body.append(_allocation(metadata))
    body.extend(
        [
            ir.TensorStore(metadata, [0], 1),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[view.ty])),
        ]
    )
    if mapping == "layout_map":
        body.append(_annotation({"layout_map": {view: Layout([8], lambda i: [i % 2, i // 2])}}))
    return params, body, output, storage, metadata, view


@pytest.mark.parametrize("mapping", ["layout_map", "fragment_scope"])
@pytest.mark.parametrize("metadata_scope", ["global", "local.fragment"])
def test_physical_view_remaps_mutable_type_operands_and_reads_them_each_time(mapping, metadata_scope):
    params, body, _, _, metadata, view = _input(mapping, metadata_scope)
    for index, value in enumerate((1, 2, 1)):
        body.append(ir.TensorStore(metadata, [0], value))
        body.append(ir.TensorStore(params[0], [index], ir.TensorLoad(view, [2])))
    executable = _compile(tirx.Function(params, body))
    tensors = [tvm_tirx.runtime.tensor(np.full(3, -99, dtype="int32"))]
    if mapping == "layout_map":
        tensors.append(tvm_tirx.runtime.tensor(np.arange(32, dtype="int32") + 10))
    if metadata_scope == "global":
        tensors.append(tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32")))
    executable(*tensors)
    np.testing.assert_array_equal(
        tensors[0].numpy(),
        np.array([12, 13, 12] if mapping == "layout_map" else [13, 14, 13], dtype="int32"),
    )


@pytest.mark.parametrize("mapping", ["layout_map", "fragment_scope"])
@pytest.mark.parametrize("consumer", ["tensor_region", "access_ptr", "ptr_access"])
def test_physical_view_uses_keep_definitions_for_every_type_operand(mapping, consumer):
    params, body, _, _, metadata, view = _input(mapping, "local.fragment")
    body.append(ir.TensorStore(metadata, [0], 2))
    if consumer == "tensor_region":
        value = ir.TensorRegion(view, [ir.Range(2, 3)], ir.TensorRegionType())
    elif consumer == "access_ptr":
        value = ir.Call("tl.access_ptr", [ir.TensorLoad(view, [2]), 1, 1])
    else:
        value = ir.Call("tl.ptr_access", [view.data, 2, 1, 1])
    body.append(ir.Evaluate(value))
    module, _ = _lower(tirx.Function(params, body))
    assert not tirx.analysis.undefined_vars(module["main"])
