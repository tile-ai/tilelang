"""Ascend disjointness follows allocation provenance through every address path."""

import pytest
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_dependencies import _conflict, _ids, _inspect, _pairs
from test_native_ascend_ops import subregion
from test_native_ascend_schedule_analysis import _api
from test_native_ascend_storage_provenance import _compare
from test_native_reducer_contracts import allocate, integer, region, tensor

from tvm_tirx import ir

ON_CHIP_SCOPES = [
    "local",
    "local.var",
    "shared",
    "shared.dyn",
    "shared.l1",
    "shared.l1.dyn",
    "shared.l0a",
    "shared.l0a.dyn",
    "shared.l0b",
    "shared.l0b.dyn",
    "shared.l0c",
    "shared.l0c.dyn",
    "shared.ss",
    "shared.ss.dyn",
]


def _declare(view, pointer):
    return ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))


def _reinterpret(pointer, scope):
    return ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType("int32", scope))


def _assert_dependency(params, definitions, writer_tensor, read_value, expected):
    writer = ir.TensorStore(writer_tensor, [integer(0)], integer(1))
    reader = ir.Evaluate(read_value)
    result = _inspect(_kernel(params, [*map(_unit, definitions), _unit(writer), _unit(reader)]))
    pair = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:2] == pair for row in _pairs(result, root_only=True)) == expected
    return result


@pytest.mark.parametrize("scope", ON_CHIP_SCOPES)
@pytest.mark.parametrize("external_kind", ["tensor", "pointer"])
def test_global_and_fresh_on_chip_regions_have_disjoint_storage(scope, external_kind):
    incoming = tensor("incoming", [4], "int32", "global")
    scratch = tensor("scratch", [4], "int32", scope)
    definitions = [allocate(scratch)]
    params = [incoming]
    if external_kind == "pointer":
        pointer = ir.Var("external", ir.PtrType("int32", "global"))
        definitions.append(_declare(incoming, pointer))
        params = [pointer]
    assert _compare(definitions, incoming, scratch)["disjoint"]
    assert _compare(definitions, scratch, incoming)["disjoint"]
    assert not _conflict(region(incoming), region(scratch), definitions)
    _assert_dependency(params, definitions, scratch, ir.TensorLoad(incoming, [integer(0)]), False)


@pytest.mark.parametrize(
    "left_scope,right_scope,right_fresh",
    [
        ("global", "global", False),
        ("global", "global", True),
        ("shared", "shared.dyn", True),
        ("global", "shared", False),
        ("global", "shared.tmem", True),
        ("global", "local.fragment", True),
    ],
)
def test_external_roots_and_unrecognized_spaces_do_not_gain_noalias(left_scope, right_scope, right_fresh):
    left = tensor("left", [4], "int32", left_scope)
    right = tensor("right", [4], "int32", right_scope)
    definitions = [allocate(right)] if right_fresh else []
    params = [left] if right_fresh else [left, right]
    assert not _compare(definitions, left, right)["disjoint"]
    assert _conflict(region(left), region(right), definitions)
    _assert_dependency(params, definitions, left, ir.TensorLoad(right, [integer(0)]), True)


@pytest.mark.parametrize("base_scope,view_scope", [("shared", "shared.dyn"), ("shared.dyn", "shared")])
def test_same_space_views_keep_origin_and_physical_byte_offset(base_scope, view_scope):
    incoming = tensor("incoming", [8], "int32", "global")
    base = tensor("base", [8], "int32", base_scope)
    view = tensor("view", [4], "int32", view_scope)
    pointer = ir.Call("tirx.ptr_byte_offset", [base.data, integer(4, "int64")], ty=base.data.ty)
    definitions = [allocate(base), _declare(view, pointer)]
    assert _compare(definitions, incoming, view)["disjoint"]
    assert _compare(definitions, base, view)["same_storage"]
    assert not _compare(definitions, base, view)["disjoint"]
    (row,) = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view], [[integer(0)]])
    assert row["exact"] and row["root"].same_as(base)
    assert int(row["bits"]) == 32
    assert _conflict(subregion(base, [1], [1]), subregion(view, [0], [1]), definitions)
    assert not _conflict(subregion(base, [0], [1]), subregion(view, [0], [1]), definitions)


@pytest.mark.parametrize("origin", ["global_parameter", "fresh_shared"])
@pytest.mark.parametrize("path", ["declaration", "direct_cast", "bound_cast", "round_trip"])
def test_scope_changing_views_cannot_recover_allocation_provenance(origin, path):
    source_scope = "global" if origin == "global_parameter" else "shared"
    other_scope = "shared" if source_scope == "global" else "global"
    source = tensor("source", [4], "int32", source_scope)
    peer = tensor("peer", [4], "int32", other_scope)
    definitions = [allocate(peer)]
    params = [source]
    if origin == "fresh_shared":
        definitions.append(allocate(source))
        params = []
    pointer = source.data
    final_scope = other_scope
    if path != "declaration":
        pointer = _reinterpret(pointer, other_scope)
    if path == "bound_cast":
        bound = ir.Var("changed_scope", pointer.ty)
        definitions.append(ir.Bind(bound, pointer))
        pointer = bound
    if path == "round_trip":
        pointer = _reinterpret(pointer, source_scope)
        final_scope = source_scope
    view = tensor("view", [4], "int32", final_scope)
    definitions.append(_declare(view, pointer))
    (row,) = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view], [[integer(0)]])
    assert not row["exact"] and "bits" not in row
    assert not _compare(definitions, peer, view)["disjoint"]
    assert _conflict(region(peer), region(view), definitions)
    _assert_dependency(params, definitions, peer, ir.TensorLoad(view, [integer(0)]), True)


def _opaque(pointer):
    return ir.Call("tirx.call_extern", [ir.StringImm("opaque_consumer"), pointer], ty="int32")


@pytest.mark.parametrize("path", ["data", "bound", "raw_access", "region_access"])
@pytest.mark.parametrize("writer_local", [False, True])
def test_opaque_accesses_use_the_same_proven_global_allocation_boundary(path, writer_local):
    incoming = tensor("incoming", [4], "int32", "global")
    scratch = tensor("scratch", [4], "int32", "shared")
    definitions = [allocate(scratch)]
    address_tensor = incoming if writer_local else scratch
    writer = scratch if writer_local else incoming
    pointer = address_tensor.data
    if path == "bound":
        bound = ir.Var("pointer", pointer.ty)
        definitions.append(ir.Bind(bound, pointer))
        pointer = bound
    elif path == "raw_access":
        pointer = ir.Call("tl.ptr_access", [pointer, integer(0), integer(1), integer(1)])
    elif path == "region_access":
        pointer = ir.Call("tl.access_ptr", [ir.TensorLoad(address_tensor, [integer(0)]), integer(1), integer(1)])
    _assert_dependency([incoming], definitions, writer, _opaque(pointer), False)


@pytest.mark.parametrize("origin", ["global_parameter", "fresh_shared"])
@pytest.mark.parametrize("path", ["direct_cast", "bound_cast", "round_trip", "raw_access", "unknown_call"])
def test_opaque_root_canonicalization_cannot_erase_a_scope_change(origin, path):
    source_scope = "global" if origin == "global_parameter" else "shared"
    peer_scope = "shared" if source_scope == "global" else "global"
    source = tensor("source", [4], "int32", source_scope)
    peer = tensor("peer", [4], "int32", peer_scope)
    params = [source]
    definitions = [allocate(peer)]
    if origin == "fresh_shared":
        params = []
        definitions.append(allocate(source))
    pointer = _reinterpret(source.data, peer_scope)
    if path == "bound_cast":
        bound = ir.Var("pointer", pointer.ty)
        definitions.append(ir.Bind(bound, pointer))
        pointer = bound
    elif path == "round_trip":
        pointer = _reinterpret(pointer, source_scope)
    elif path == "raw_access":
        pointer = ir.Call("tl.ptr_access", [pointer, integer(0), integer(1), integer(1)])
    elif path == "unknown_call":
        pointer = ir.Call("tirx.call_extern", [ir.StringImm("arbitrary_pointer"), source.data], ty=source.data.ty)
    _assert_dependency(params, definitions, peer, _opaque(pointer), True)


@pytest.mark.parametrize("origin", ["integer", "unknown_call", "global_parameter"])
def test_unproven_opaque_pointer_is_conservative_against_fresh_local_storage(origin):
    scratch = tensor("scratch", [4], "int32", "shared")
    pointer_type = ir.PtrType("int32", "global")
    definitions = [allocate(scratch)]
    if origin == "integer":
        parameter = ir.Var("address", "uint64")
        pointer = ir.Call("tirx.reinterpret", [parameter], ty=pointer_type)
        params = [parameter]
    elif origin == "unknown_call":
        pointer = ir.Var("pointer", pointer_type)
        definitions.append(
            ir.Bind(
                pointer,
                ir.Call("tirx.call_extern", [ir.StringImm("arbitrary_pointer")], ty=pointer_type),
            )
        )
        params = []
    else:
        pointer = ir.Var("external", pointer_type)
        params = [pointer]
    _assert_dependency(params, definitions, scratch, _opaque(pointer), origin != "global_parameter")


@pytest.mark.parametrize("origin", ["constant", "rootless_call", "offset", "loaded_address"])
def test_rootless_pointer_effects_remain_in_the_dependency_graph(origin):
    scratch = tensor("scratch", [4], "int32", "shared")
    pointer_type = ir.PtrType("int32", "global")
    params = []
    if origin == "rootless_call":
        pointer = ir.Call("tirx.call_extern", [ir.StringImm("arbitrary_pointer")], ty=pointer_type)
    else:
        if origin == "loaded_address":
            addresses = tensor("addresses", [1], "uint64", "global")
            params = [addresses]
            address = ir.TensorLoad(addresses, [integer(0)])
        else:
            address = integer(4096, "uint64")
        pointer = ir.Call("tirx.reinterpret", [address], ty=pointer_type)
        if origin == "offset":
            pointer = ir.Call("tirx.ptr_byte_offset", [pointer, integer(16, "int64")], ty=pointer_type)
    _assert_dependency(params, [allocate(scratch)], scratch, _opaque(pointer), True)


@pytest.mark.parametrize("origin", ["integer", "rootless_call"])
def test_unresolved_pointer_definition_does_not_access_its_pointee(origin):
    scratch = tensor("scratch", [4], "int32", "shared")
    pointer_type = ir.PtrType("int32", "global")
    params = []
    if origin == "integer":
        address = ir.Var("address", "uint64")
        params = [address]
        pointer = ir.Call("tirx.reinterpret", [address], ty=pointer_type)
    else:
        pointer = ir.Call("tirx.call_extern", [ir.StringImm("arbitrary_pointer")], ty=pointer_type)
    writer = ir.TensorStore(scratch, [integer(0)], integer(1))
    bound = ir.Var("pointer", pointer_type)
    definition = ir.Bind(bound, pointer)
    result = _inspect(_kernel(params, [_unit(allocate(scratch)), _unit(writer), _unit(definition)]))
    pair = (_ids(result, writer)[0], _ids(result, bound)[0])
    assert not any(row[:2] == pair for row in _pairs(result, root_only=True))


@pytest.mark.parametrize("result_kind", ["pointer", "tensor"])
@pytest.mark.parametrize("argument", ["scope_change", "rootless"])
def test_pointer_producing_call_keeps_unknown_pointer_argument_effects(result_kind, argument):
    incoming = tensor("incoming", [4], "int32", "global")
    scratch = tensor("scratch", [4], "int32", "shared")
    params = [incoming]
    if argument == "scope_change":
        pointer = _reinterpret(incoming.data, "shared")
    else:
        pointer = ir.Call("tirx.reinterpret", [integer(4096, "uint64")], ty=ir.PtrType("int32", "global"))
    value = ir.Call("tirx.call_extern", [ir.StringImm("pointer_producer"), pointer], ty=pointer.ty)
    if result_kind == "pointer":
        bound = ir.Var("produced_pointer", pointer.ty)
        definition = ir.Bind(bound, value)
    else:
        bound = tensor("produced_tensor", [4], "int32", pointer.ty.scope.name)
        definition = _declare(bound, value)
    writer = ir.TensorStore(scratch, [integer(0)], integer(1))
    result = _inspect(_kernel(params, [_unit(allocate(scratch)), _unit(writer), _unit(definition)]))
    pair = (_ids(result, writer)[0], _ids(result, bound)[0])
    assert any(row[:2] == pair for row in _pairs(result, root_only=True))


def test_aiv_scope_labels_do_not_restore_a_scope_changing_views_provenance():
    incoming = tensor("incoming", [4], "int32", "global")
    view = tensor("view", [4], "int32", "shared")
    scratch = tensor("scratch", [4], "int32", "shared")
    writer = ir.TensorStore(view, [integer(0)], integer(1))
    reader = ir.Evaluate(ir.TensorLoad(scratch, [integer(0)]))
    result = _inspect(
        _kernel(
            [incoming],
            [
                _unit(allocate(scratch)),
                _unit(_declare(view, _reinterpret(incoming.data, "shared"))),
                _unit(writer, core=2, sid=0),
                _unit(reader, core=4, sid=1),
            ],
        )
    )
    pair = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:2] == pair for row in _pairs(result, root_only=True))
