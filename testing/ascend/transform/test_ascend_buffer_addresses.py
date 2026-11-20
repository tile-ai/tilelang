"""Exact placement of physical on-chip allocations after version/layout lowering."""

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes, seq


SCOPES = [
    pytest.param("shared.dyn", 32, 248 * 1024, id="ub"),
    pytest.param("shared.l1", 512, 512 * 1024, id="l1"),
    pytest.param("shared.l0a", 512, 64 * 1024, id="l0a"),
    pytest.param("shared.l0b", 512, 64 * 1024, id="l0b"),
    pytest.param("shared.l0c", 1024, 256 * 1024, id="l0c"),
    pytest.param("shared.ss", 32, 3 * 1024, id="ss"),
]


def _allocation_program(buffers, addresses=None, *, aliases=True, aiv_storages=None, simt=False):
    # Input to the allocator already has physical versions and flattened sizes.
    # Frontend version-count validation belongs to AutoSchedule.
    stores = seq(*[tirx.BufferStore(buffer, tirx.const(index, buffer.dtype), [0]) for index, buffer in enumerate(buffers)])
    if simt:
        stores = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "SIMT_VF", stores))
    body = tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in buffers], stores])
    if aliases:
        # Even an explicit positive alias permission cannot authorize overlap.
        body = tirx.AttrStmt(
            {buffer.data: [other.data for other in buffers if other != buffer] for buffer in buffers}, "tl.buffer_alias_map", 1, body
        )
    if addresses is not None:
        body = tirx.AttrStmt(addresses, "tl.ascend_buffer_addresses", 1, body)
    if aiv_storages is not None:
        body = tirx.AttrStmt(aiv_storages, "tl.aiv_storage_map", 1, body)
    thread = tirx.IterVar(None, tirx.Var("bx", "int32"), tirx.IterVar.ThreadIndex, "blockIdx.x")
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt(thread, "thread_extent", 1, body))})


def _buffer(name, size=512, scope="shared.l1"):
    return tirx.decl_buffer((size // 4,), "uint32", name=name, scope=scope)


def _store_offsets(after):
    return [int(tvm.arith.Analyzer().simplify(store.indices[0])) * 4 for store in nodes(after, tirx.BufferStore)]


@pytest.mark.parametrize("scope,alignment,capacity", SCOPES)
@pytest.mark.parametrize("disable_reuse", [False, True])
def test_exact_plan_keeps_holes_and_physical_addresses(scope, alignment, capacity, disable_reuse):
    a, b = _buffer("a", alignment, scope), _buffer("b", 2 * alignment, scope)
    before = _allocation_program([a, b], {a.data: 4 * alignment, b.data: 0})
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(before)
    assert _store_offsets(after) == [4 * alignment, 0]
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert allocation.buffer.dtype == "uint8"
    assert int(allocation.buffer.shape[0]) == 5 * alignment
    assert not any(node.attr_key == "tl.ascend_buffer_addresses" for node in nodes(after, tirx.AttrStmt))


@pytest.mark.parametrize("scope,alignment,capacity", SCOPES)
def test_single_allocation_can_end_at_capacity(scope, alignment, capacity):
    a = _buffer("a", alignment, scope)
    after = transform.MergeUBAllocations()(_allocation_program([a], {a.data: capacity - alignment}))
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert int(allocation.buffer.shape[0]) == capacity
    assert _store_offsets(after) == [capacity - alignment]


@pytest.mark.parametrize("scope,alignment,capacity", SCOPES)
def test_partial_plan_first_fits_unplaced_buffers_without_reuse(scope, alignment, capacity):
    large, fixed, small = [_buffer(name, size * alignment, scope) for name, size in [("large", 2), ("fixed", 1), ("small", 1)]]
    after = transform.MergeUBAllocations()(_allocation_program([large, fixed, small], {fixed.data: alignment}))
    assert _store_offsets(after) == [2 * alignment, alignment, 0]
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert int(allocation.buffer.shape[0]) == 4 * alignment


@pytest.mark.parametrize("scope,alignment,capacity", SCOPES)
@pytest.mark.parametrize("case", ["overlap", "capacity", "unaligned"])
def test_scope_specific_physical_constraints(scope, alignment, capacity, case):
    a, b = _buffer("a", 2 * alignment, scope), _buffer("b", alignment, scope)
    addresses = {a.data: 0, b.data: alignment}
    message = "overlap"
    if case == "capacity":
        addresses = {a.data: capacity - alignment}
        message = "capacity"
    elif case == "unaligned":
        addresses = {a.data: alignment // 2}
        message = "aligned integer addresses"
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.MergeUBAllocations()(_allocation_program([a, b], addresses))


@pytest.mark.parametrize(
    "case,message",
    [
        ("unknown", "unknown allocation"),
        ("scope", "does not support storage scope"),
        ("negative", "aligned integer addresses"),
        ("boolean", "integer addresses"),
        ("overflow", "capacity"),
        ("dynamic", "positive constant"),
        ("empty", "must not be empty"),
        ("duplicate", "one allocation per storage"),
        ("no-contract", "scheduled allocation"),
    ],
)
def test_invalid_physical_plans_fail(case, message):
    a, b = _buffer("a"), _buffer("b")
    if case == "dynamic":
        a = tirx.decl_buffer((tirx.Var("n", "int32"),), "uint32", name="a", scope="shared.l1")
    elif case == "overflow":
        a = _buffer("a", 2**62)
    elif case == "scope":
        a = _buffer("a", scope="local")
    addresses = {a.data: 0, b.data: 1024}
    if case == "unknown":
        addresses[_buffer("unknown").data] = 2048
    elif case == "negative":
        addresses[a.data] = -512
    elif case == "boolean":
        addresses[a.data] = tirx.const(True, "bool")
    elif case == "empty":
        addresses = {}
    buffers = [a, b, a] if case == "duplicate" else [a, b]
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.MergeUBAllocations()(_allocation_program(buffers, addresses, aliases=case != "no-contract"))


def test_dead_allocation_cleanup_preserves_invalid_unknown_placement():
    a, unknown = _buffer("a"), _buffer("unknown")
    before = _allocation_program([a], {unknown.data: 512})
    after = transform.AscendRemoveNoOp()(before)
    with pytest.raises(tvm.error.InternalError, match="unknown allocation"):
        transform.MergeUBAllocations()(after)


def test_existing_dynamic_arena_is_rejected():
    a = _buffer("a")
    arena = _buffer("arena", scope="shared.l1.dyn")
    with pytest.raises(tvm.error.InternalError, match="existing shared.l1.dyn arena"):
        transform.MergeUBAllocations()(_allocation_program([a, arena], {a.data: 0}))


@pytest.mark.parametrize("simt", [False, True])
def test_ub_placement_accounts_for_simt_context(simt):
    a = _buffer("a", 64, "shared.dyn")
    before = _allocation_program([a], {a.data: 216 * 1024 - 32}, simt=simt)
    if simt:
        with pytest.raises(tvm.error.InternalError, match="capacity of 221184 bytes"):
            transform.MergeUBAllocations()(before)
    else:
        assert _store_offsets(transform.MergeUBAllocations()(before)) == [216 * 1024 - 32]


@pytest.mark.parametrize("conflict", [None, "same-aiv", "common"])
def test_ub_overlap_uses_physical_aiv_address_spaces(conflict):
    common, v0, v1 = [_buffer(name, 32, "shared.dyn") for name in ("common", "v0", "v1")]
    addresses = {v0.data: 0, v1.data: 0}
    if conflict == "common":
        addresses[common.data] = 0
    before = _allocation_program([common, v0, v1], addresses, aiv_storages={v0.data: 0, v1.data: 0 if conflict == "same-aiv" else 1})
    if conflict:
        with pytest.raises(tvm.error.InternalError, match="overlap"):
            transform.MergeUBAllocations()(before)
    else:
        assert _store_offsets(transform.MergeUBAllocations()(before)) == [32, 0, 0]


def test_different_pools_use_independent_addresses():
    buffers = [_buffer(str(i), alignment, scope) for i, (scope, alignment, _) in enumerate(item.values for item in SCOPES)]
    before = _allocation_program(buffers, {buffer.data: 1024 for buffer in buffers})
    after = transform.MergeUBAllocations()(before)
    assert _store_offsets(after) == [1024] * len(buffers)
    assert len(nodes(after, tirx.AllocBuffer)) == len(buffers)


@pytest.mark.parametrize("address", [True, -512, 1, 1.5, 524288, 2**64])
def test_frontend_rejects_invalid_addresses(address):
    with pytest.raises(ValueError, match="byte address"):
        T.annotate_buffer_addresses({_buffer("a"): address})


@pytest.mark.parametrize("scope", ["shared", "global", "local.fragment", "shared.l0a.sf"])
def test_frontend_rejects_scopes_without_independent_placement(scope):
    with pytest.raises(ValueError, match="does not support storage scope"):
        T.annotate_buffer_addresses({_buffer("a", scope=scope): 0})


def test_frontend_aliases_share_one_storage_address():
    a = _buffer("a")
    view = tirx.decl_buffer((512,), "uint8", data=a.data, name="view", scope="shared.l1")
    with pytest.raises(ValueError, match="same byte address"):
        T.annotate_buffer_addresses({a: 0, view: 512})


def test_absent_placement_preserves_existing_no_contract_behavior():
    before = _allocation_program([_buffer("a")], aliases=False)
    tvm.ir.assert_structural_equal(before, transform.MergeUBAllocations()(before))
