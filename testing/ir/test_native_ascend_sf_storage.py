"""Scale-factor allocations have known on-chip provenance without erasing casts."""

import pytest
import test_native_ascend_storage_spaces as spaces
from test_native_ascend_dependencies import _conflict
from test_native_ascend_schedule_analysis import _api
from test_native_ascend_storage_provenance import _compare
from test_native_reducer_contracts import allocate, integer, region, tensor

from tvm_tirx import ir


@pytest.mark.parametrize("scope", ["shared.l0a.sf", "shared.l0b.sf"])
@pytest.mark.parametrize("external_kind", ["tensor", "pointer"])
def test_fresh_scale_factor_storage_is_disjoint_from_global(scope, external_kind):
    spaces.test_global_and_fresh_on_chip_regions_have_disjoint_storage(scope, external_kind)


@pytest.mark.parametrize("scope", ["shared.l0a.sf", "shared.l0b.sf"])
def test_scale_factor_parameter_does_not_gain_allocation_provenance(scope):
    spaces.test_external_roots_and_unrecognized_spaces_do_not_gain_noalias("global", scope, False)


@pytest.mark.parametrize(
    "origin_scope,view_scope",
    [
        ("global", "shared.l0a.sf"),
        ("global", "shared.l0b.sf"),
        ("shared.l0a", "shared.l0a.sf"),
        ("shared.l0b", "shared.l0b.sf"),
        ("shared.l0a.sf", "shared.l0b.sf"),
    ],
)
@pytest.mark.parametrize("roundtrip", [False, True])
def test_scale_factor_scope_changes_keep_opaque_provenance(origin_scope, view_scope, roundtrip):
    origin = tensor("origin", [4], "int32", origin_scope)
    peer = tensor("peer", [4], "int32", "global")
    definitions = [] if origin_scope == "global" else [allocate(origin)]
    params = [origin, peer] if origin_scope == "global" else [peer]
    pointer = spaces._reinterpret(origin.data, view_scope)
    if roundtrip:
        pointer = spaces._reinterpret(pointer, origin_scope)
        view_scope = origin_scope
    view = tensor("view", [4], "int32", view_scope)
    definitions.append(spaces._declare(view, pointer))
    (address,) = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view], [[integer(0)]])
    assert not address["exact"] and "bits" not in address
    assert not _compare(definitions, peer, view)["disjoint"]
    assert _conflict(region(peer), region(view), definitions)
    spaces._assert_dependency(params, definitions, peer, ir.TensorLoad(view, [integer(0)]), True)


@pytest.mark.parametrize("scope", ["shared.l0a.sf", "shared.l0b.sf"])
@pytest.mark.parametrize("origin", ["integer", "external-call"])
def test_opaque_global_pointer_can_alias_scale_factor_storage(scope, origin):
    scratch = tensor("scratch", [4], "int32", scope)
    pointer_type = ir.PtrType("int32", "global")
    if origin == "integer":
        source = ir.Var("address", "uint64")
        pointer = ir.Call("tirx.reinterpret", [source], ty=pointer_type)
        params = [source]
    else:
        pointer = ir.Call("tirx.call_extern", [ir.StringImm("unknown_pointer")], ty=pointer_type)
        params = []
    spaces._assert_dependency(params, [allocate(scratch)], scratch, spaces._opaque(pointer), True)
