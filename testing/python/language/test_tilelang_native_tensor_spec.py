"""Tensor annotation constraints survive materialization into native IR."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.language._native import buffer, tensor_offset, tensor_strides


def decode(node):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])


def test_offset_factor_materializes_symbolic_offset_and_typed_entry_contract():
    spec = buffer((8,), "float32", offset_factor=8)
    assert isinstance(spec.elem_offset, ir.Var)

    @T.prim_func
    def kernel(A: spec):
        A[0] = 1.0

    assert tirx.analysis.check_well_formed(kernel)
    entry = kernel.body.seq[0]
    assert isinstance(entry, ir.RegionStmt) and entry.op.name == "tl.attr"
    contract = decode(entry)
    assert contract["key"] == "tl.tensor_spec"
    parameter = kernel.params[0]
    assert contract["value"][parameter]["offset_factor"] == 8
    assert tensor_offset(parameter).same_as(spec.elem_offset)
    # This declaration is an ABI obligation; it does not itself prove that a
    # supplied runtime tensor has a suitably aligned byte_offset.
    assert len(entry.args[0].fields) == 1 and entry.args[0].fields[0].same_as(parameter)


def test_auto_broadcast_preserves_physical_stride_parameters_for_abi_consumer():
    n = T.dynamic("n")
    spec = buffer((n, 1), "float32", buffer_type="auto")
    assert len(spec.strides) == 2 and all(isinstance(value, ir.Var) for value in spec.strides)

    @T.prim_func
    def kernel(A: spec):
        A[0, 0] = 1.0

    parameter = kernel.params[0]
    assert all(left.same_as(right) for left, right in zip(tensor_strides(parameter), spec.strides, strict=True))
    contract = decode(kernel.body.seq[0])["value"][parameter]
    assert contract["buffer_type"] == "auto"
    assert contract["offset_factor"] == 1
    assert tirx.analysis.check_well_formed(kernel)


def test_tensor_spec_keys_are_renewed_with_the_parameter_identity():
    spec = buffer((2, 4), "float32", elem_offset=0, offset_factor=4, axis_separators=[1])

    @T.prim_func
    def kernel(A: spec):
        A[0, 0] = 1.0

    renewed = tirx.renew_def(kernel)
    entries = decode(renewed.body.seq[0])["value"]
    assert renewed.params[0] in entries
    assert kernel.params[0] not in entries
    assert list(entries[renewed.params[0]]["axis_separators"]) == [1]
    assert tirx.analysis.check_well_formed(renewed)


def test_local_tensor_spec_is_emitted_after_its_definition():
    @T.prim_func
    def kernel():
        A = T.alloc_buffer((4,), "int32", elem_offset=0, offset_factor=4)
        A[0] = 0

    allocation, metadata, store = kernel.body.seq
    assert isinstance(allocation, ir.Bind)
    assert isinstance(metadata, ir.Evaluate) and metadata.value.op.name == "tl.annotation"
    assert isinstance(store, ir.TensorStore)
    assert decode(metadata.value)["tl.tensor_spec"][allocation.var]["offset_factor"] == 4
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("separators", [[-1], [3], [2, 1]])
def test_invalid_axis_separators_are_rejected(separators):
    with pytest.raises(ValueError, match="axis_separators"):
        buffer((2, 4), axis_separators=separators)


@pytest.mark.parametrize("separators", [[0], [2], [1, 1], [0, 0, 2]])
def test_nondecreasing_axis_separators_preserve_source_entries(separators):
    spec = buffer((2, 4), axis_separators=separators)
    assert list(spec.axis_separators) == separators
    assert spec.contract_metadata()["compact"] is True
    strided = buffer((2, 4), strides=(4, 1), axis_separators=separators)
    assert strided.contract_metadata()["compact"] is False


def test_default_contract_and_invalid_options():
    assert buffer((2, 4)).contract_metadata() is None
    assert buffer((2, 4), buffer_type="default").contract_metadata() is None
    assert buffer((2, 4), buffer_type="auto_broadcast").buffer_type == "auto"
    with pytest.raises(ValueError, match="buffer_type"):
        buffer((2, 4), buffer_type="unknown")
    with pytest.raises(ValueError, match="offset_factor"):
        buffer((2, 4), offset_factor=-1)
