"""Address arithmetic alone cannot prove that two storage roots are disjoint."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from test_native_ascend_ops import function
from test_native_ascend_schedule_analysis import _api, _has, _memory
from test_native_reducer_contracts import allocate, integer, tensor


@pytest.mark.parametrize("offset", [0, 16])
def test_integer_address_is_opaque_even_when_its_byte_offset_is_known(offset):
    address = ir.Var("address", "uint64")
    view = tensor("view", [4], "int32", "global")
    pointer_type = ir.PtrType("int32", "global")
    pointer = ir.Call("tirx.reinterpret", [address], ty=pointer_type)
    if offset:
        pointer = ir.Call("tirx.ptr_byte_offset", [pointer, integer(offset, "int64")], ty=pointer_type)
    definitions = [ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))]
    body = [ir.TensorStore(view, [integer(0)], integer(7))]
    assert tirx.analysis.check_well_formed(function([address], [*definitions, *body]))
    (row,) = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view], [[integer(0)]])
    assert not row["exact"]
    assert "bits" not in row
    assert row["root"].same_as(address)
    assert _has(_memory(body, definitions)["opaque"], address)


def _compare(definitions, left, right):
    return tvm_ffi.get_global_func("tl.testing.CompareAscendStorage")(ir.SeqStmt(definitions), left, right)


def test_distinct_external_pointer_roots_are_not_an_allocation_noalias_proof():
    left_ptr = ir.Var("pointer", ir.PtrType("int32", "global"))
    right_ptr = ir.Var("pointer", left_ptr.ty)
    left, right = tensor("left", [4], "int32", "global"), tensor("right", [4], "int32", "global")
    definitions = [
        ir.Bind(left, ir.Call("tirx.decl_tensor", [left_ptr], ty_args=[left.ty])),
        ir.Bind(right, ir.Call("tirx.decl_tensor", [right_ptr], ty_args=[right.ty])),
    ]
    assert tirx.analysis.check_well_formed(function([left_ptr, right_ptr], definitions))
    rows = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [left, right], [[integer(1)]] * 2)
    assert all(row["exact"] and not row["fresh_allocation"] for row in rows)
    assert all(int(row["bits"]) == 32 for row in rows)
    assert not _compare(definitions, left, right)["disjoint"]
    assert not _compare(definitions, left, right)["same_storage"]


def test_distinct_external_tensor_parameters_may_alias():
    left, right = tensor("left", [4], "int32", "global"), tensor("right", [4], "int32", "global")
    assert not _compare([], left, right)["disjoint"]


def test_fresh_allocations_are_disjoint_but_their_views_keep_the_same_origin():
    left, right = tensor("left", [16], "int32", "global"), tensor("right", [16], "int32", "global")
    alias = tensor("alias", [4], "int32", "global")
    pointer = ir.Call("tirx.ptr_byte_offset", [left.data, integer(16, "int64")], ty=left.data.ty)
    definitions = [allocate(left), allocate(right), ir.Bind(alias, ir.Call("tirx.decl_tensor", [pointer], ty_args=[alias.ty]))]
    assert tirx.analysis.check_well_formed(function([], definitions))
    rows = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [left, right, alias], [[integer(0)]] * 3)
    assert all(row["exact"] and row["fresh_allocation"] for row in rows)
    assert rows[2]["root"].same_as(left)
    assert int(rows[2]["bits"]) == 16 * 8
    assert _compare(definitions, left, right)["disjoint"]
    assert _compare(definitions, alias, right)["disjoint"]
    assert not _compare(definitions, left, alias)["disjoint"]
    assert _compare(definitions, left, alias)["same_storage"]


def test_reinterpreting_an_integer_does_not_gain_provenance_from_a_fresh_peer():
    address = ir.Var("address", "uint64")
    left, right = tensor("left", [4], "int32", "global"), tensor("right", [4], "int32", "global")
    pointer = ir.Call("tirx.reinterpret", [address], ty=ir.PtrType("int32", "global"))
    definitions = [allocate(left), ir.Bind(right, ir.Call("tirx.decl_tensor", [pointer], ty_args=[right.ty]))]
    assert not _compare(definitions, left, right)["disjoint"]
