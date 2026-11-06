"""A declaration owns one allocation or one pointer view at its source position."""

from tilelang.language import _tensor_spec

import inspect

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang
import tilelang.ir as tl_ir
import tilelang.language as T


def _nodes(value, node_type):
    result = []
    tvm_ffi.structural_walk(value, (node_type, result.append), order="pre")
    return result


def _producers(function):
    return [
        node
        for node in _nodes(function.body, ir.Bind)
        if isinstance(node.value, ir.Call) and node.value.op.name in ("tirx.alloc_tensor", "tirx.decl_tensor")
    ]


def _allocate_sample():
    @T.prim_func
    def sample(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        scratch = T.decl_buffer((4,), "int32", scope="local", align=16)
        for i in T.serial(4):
            scratch[i] = A[i] * 3
        for i in T.serial(4):
            B[i] = scratch[i] + 2

    return sample


def test_decl_buffer_without_data_emits_one_typed_allocation():
    function = _allocate_sample()
    assert tirx.analysis.check_well_formed(function)
    (allocation,) = _producers(function)
    assert allocation.value.op.name == "tirx.alloc_tensor"
    assert len(allocation.value.args) == 0
    assert allocation.var.name == "scratch"
    assert isinstance(allocation.var.ty, tirx.TensorType)
    assert allocation.var.ty.scope.name == "local"
    assert allocation.var.ty.dtype.dtype == "int32"
    assert allocation.var.ty.data_alignment == 16
    assert len(allocation.value.ty_args) == 1
    ir.assert_structural_equal(allocation.var.ty, allocation.value.ty_args[0])
    assert all(store.dest.same_as(allocation.var) for store in _nodes(function, ir.TensorStore)[:1])


def test_decl_buffer_source_locations_belong_to_the_emitted_definition():
    function = _allocate_sample()
    (allocation,) = _producers(function)
    lines, start = inspect.getsourcelines(_allocate_sample)
    line = start + next(i for i, text in enumerate(lines) if "scratch = T.decl_buffer" in text)
    for value in (allocation, allocation.var, allocation.value):
        assert tl_ir.span_to_location(value.loc) == (__file__, line)


def test_decl_buffer_local_allocation_executes_cpu():
    target = "c"
    kernel = tilelang.compile(_allocate_sample(), target=target, target_host=target, execution_backend="tvm_ffi")
    source = torch.tensor([-7, 0, 4, 41], dtype=torch.int32)
    output = torch.empty_like(source)
    kernel(source, output)
    np.testing.assert_array_equal(output.numpy(), [-19, 2, 14, 125])


def _view_sample():
    @T.prim_func
    def sample(A: T.Tensor((10,), "int32"), B: T.Tensor((3,), "int32")):
        alias = T.decl_buffer((3,), "int32", data=A.data, strides=(2,), elem_offset=1, align=4)
        for i in T.serial(3):
            alias[i] = alias[i] + 100
            B[i] = alias[i]

    return sample


def test_decl_buffer_explicit_pointer_emits_one_view_with_stride_and_offset():
    function = _view_sample()
    assert tirx.analysis.check_well_formed(function)
    (view,) = _producers(function)
    assert view.value.op.name == "tirx.decl_tensor"
    assert view.var.name == "alias"
    assert len(view.value.args) == 1
    assert isinstance(view.value.args[0], ir.Call)
    assert view.value.args[0].op.name == "tirx.tensor_data_ptr"
    assert view.value.args[0].args[0].same_as(function.params[0])
    assert int(_tensor_spec.tensor_strides(view.var)[0]) == 2
    assert int(_tensor_spec.tensor_offset(view.var)) == 1
    clone = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, clone, map_free_vars=True)
    assert tirx.analysis.check_well_formed(clone)


def test_decl_buffer_pointer_view_updates_the_original_storage_cpu():
    target = "c"
    kernel = tilelang.compile(_view_sample(), target=target, target_host=target, execution_backend="tvm_ffi")
    source = torch.arange(10, dtype=torch.int32)
    output = torch.empty(3, dtype=torch.int32)
    kernel(source, output)
    np.testing.assert_array_equal(output.numpy(), [101, 103, 105])
    np.testing.assert_array_equal(source.numpy(), [0, 101, 2, 103, 4, 105, 6, 7, 8, 9])


def test_decl_buffer_tensor_spec_metadata_uses_the_defined_tensor():
    @T.prim_func
    def sample(A: T.Tensor((8,), "int32"), B: T.Tensor((2, 2), "int32")):
        alias = T.decl_buffer(
            (2, 2),
            "int32",
            data=A.data,
            strides=(4, 1),
            elem_offset=0,
            offset_factor=2,
            axis_separators=(1,),
            align=4,
        )
        for i, j in T.grid(2, 2):
            B[i, j] = alias[i, j]

    assert tirx.analysis.check_well_formed(sample)
    (view,) = _producers(sample)
    metadata = [call for call in _nodes(sample, ir.Call) if call.op.name == "tl.annotation"]
    assert len(metadata) == 1
    payload = tvm_ffi.get_global_func("tl.DecodeAnnotations")(metadata[0].args[0], metadata[0].attrs["annotation_schema"])
    (tensor,) = payload["tl.tensor_spec"].keys()
    assert tensor.same_as(view.var)
    assert payload["tl.tensor_spec"][tensor]["offset_factor"] == 2
    assert list(payload["tl.tensor_spec"][tensor]["axis_separators"]) == [1]


def test_decl_buffer_is_emitted_in_its_lexical_branch():
    @T.prim_func
    def sample(flag: T.int32, B: T.Tensor((1,), "int32")):
        if flag:
            scratch = T.decl_buffer((1,), "int32", scope="local")
            scratch[0] = 9
            B[0] = scratch[0]
        else:
            scratch = T.decl_buffer((1,), "int32", scope="local")
            scratch[0] = 4
            B[0] = scratch[0]

    assert tirx.analysis.check_well_formed(sample)
    (branch,) = _nodes(sample, ir.If)
    then_alloc = _producers(tirx.Function([], branch.then_case, ret_type=ir.PrimType("void")))
    else_alloc = _producers(tirx.Function([], branch.else_case, ret_type=ir.PrimType("void")))
    assert len(then_alloc) == len(else_alloc) == 1
    assert not then_alloc[0].var.same_as(else_alloc[0].var)


def test_decl_buffer_requires_a_builder_while_tensor_stays_an_annotation():
    annotation = T.Tensor((4,), "float32")
    assert isinstance(annotation, _tensor_spec.TensorSpec)
    assert not IRBuilder.is_in_scope()
    with pytest.raises(ValueError, match="No builder in current scope"):
        T.decl_buffer((4,), "float32")
