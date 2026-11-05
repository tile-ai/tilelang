"""Storage reuse follows complete definitions and proven completion order."""

import pytest
from tvm_tirx import ir

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_dependencies import _api, _ids, _inspect, _root
from test_native_reducer_contracts import allocate, integer, tensor


def _write(storage, value=1):
    return ir.TensorStore(storage, [integer(0)], integer(value))


def _read(storage):
    return ir.Evaluate(ir.TensorLoad(storage, [integer(0)]))


def _aliases(value, edges, *, clock=None, periods=None, recurrence=False, implicit=False):
    inspected = _inspect(value)
    numeric = [[_ids(inspected, lhs)[0], _ids(inspected, rhs)[0], distance, int(strong)] for lhs, rhs, distance, strong in edges]
    root = _root(value)
    before = ir.save_json(root)
    result = _api("AnalyzeAscendLifetimes")(root, clock, numeric, periods or {}, implicit, recurrence)
    assert ir.save_json(root) == before
    for storage, peers in result.items():
        assert all(not peer.same_as(storage) and any(reverse.same_as(storage) for reverse in result[peer]) for peer in peers)
    return result


def _can_alias(aliases, lhs, rhs):
    return any(candidate.same_as(rhs) for candidate in aliases.get(lhs, []))


@pytest.mark.parametrize(
    "ordering,expected", [("complete", True), ("weak", False), ("missing", False), ("overlap", False), ("cycle", False)]
)
def test_live_ranges_need_completion_order_for_every_endpoint(ordering, expected):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    wa, ra, wb, rb = _write(a), _read(a), _write(b), _read(b)
    sequence = [wa, ra, wb, rb] if ordering != "overlap" else [wa, wb, ra, rb]
    edges = [(lhs, rhs, 0, True) for lhs, rhs in zip(sequence, sequence[1:])]
    if ordering == "weak":
        edges[1] = (ra, wb, 0, False)
    elif ordering == "missing":
        edges.pop(1)
    elif ordering == "cycle":
        edges.append((rb, wa, 0, True))
    value = _kernel([], [_unit(allocate(a)), _unit(allocate(b)), *map(_unit, sequence)])
    assert _can_alias(_aliases(value, edges), a, b) == expected


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("break_kind", [None, "for", "while"])
def test_partial_or_early_exit_writes_do_not_initialize_a_later_read(guarded, break_kind):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    enabled, index = ir.Var("enabled", "bool"), ir.Var("j", "int32")
    body = [_write(a)]
    if guarded:
        body = [ir.If(enabled, body, None)]
    if break_kind:
        body = [ir.If(enabled, [ir.Break()], None), *body]
    if break_kind == "while":
        body.append(ir.Break())
        atomic = ir.While(integer(True, "bool"), body)
    else:
        atomic = ir.For(index, integer(0), integer(1), integer(1), ir.ForKind.DEFAULT, body)
    writer = ir.SeqStmt([atomic])
    ra, wb, rb = _read(a), _write(b), _read(b)
    value = _kernel([enabled], [_unit(allocate(a)), _unit(allocate(b)), _unit(writer), _unit(ra), _unit(wb), _unit(rb)])
    edges = [(atomic, ra, 0, True), (ra, wb, 0, True), (wb, rb, 0, True)]
    assert _can_alias(_aliases(value, edges), a, b) == (not guarded and break_kind is None)


@pytest.mark.parametrize("branch_form,mutable", [("single", False), ("single", True), ("separate", False), ("separate", True)])
def test_lexical_exclusion_uses_the_actual_condition_capture(branch_form, mutable):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    flag = tensor("flag", [1], "int32", "global") if mutable else ir.Var("flag", "bool")
    condition = ir.TensorLoad(flag, [integer(0)]) > 0 if mutable else flag
    wa, ra, wb, rb = _write(a), _read(a), _write(b), _read(b)
    if branch_form == "single":
        body = [ir.If(condition, [_unit(wa), _unit(ra)], [_unit(wb), _unit(rb)])]
    else:
        body = [ir.If(condition, [_unit(wa), _unit(ra)], None), ir.If(ir.prim.Not(condition), [_unit(wb), _unit(rb)], None)]
    value = _kernel([flag], [_unit(allocate(a)), _unit(allocate(b)), *body])
    aliases = _aliases(value, [(wa, ra, 0, True), (wb, rb, 0, True)])
    assert _can_alias(aliases, a, b) == (branch_form == "single" or not mutable)


@pytest.mark.parametrize("predicate_kind", ["parameter", "changing_load", "outer_snapshot", "inner_snapshot", "iteration"])
@pytest.mark.parametrize("step", [1, 2])
def test_periodic_exclusion_requires_a_guard_stable_between_iterations(predicate_kind, step):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    flag = tensor("flag", [1], "int32", "global")
    enabled, index, snapshot = ir.Var("enabled", "bool"), ir.Var("i", "int32"), ir.Var("snapshot", "bool")
    load = ir.TensorLoad(flag, [integer(0)]) > 0
    before, prefix = [], []
    condition = enabled
    if predicate_kind == "changing_load":
        condition = load
    elif predicate_kind in {"outer_snapshot", "inner_snapshot"}:
        condition = snapshot
        (before if predicate_kind == "outer_snapshot" else prefix).append(_unit(ir.Bind(snapshot, load)))
    elif predicate_kind == "iteration":
        condition = (index // step) % 2 == 0
    wa, ra, wb, rb = _write(a), _read(a), _write(b), _read(b)
    body = [*prefix, ir.If(condition, [_unit(wa), _unit(ra)], [_unit(wb), _unit(rb)])]
    loop = _loop(index, body, begin=0, end=4 * step, step=step)
    value = _kernel([flag, enabled], [_unit(allocate(a)), _unit(allocate(b)), *before, loop])
    aliases = _aliases(value, [(wa, ra, 0, True), (wb, rb, 0, True)], clock=index, periods={a: 1, b: 1}, recurrence=True)
    assert _can_alias(aliases, a, b) == (predicate_kind in {"parameter", "outer_snapshot"})


@pytest.mark.parametrize("origin", ["parameter", "integer", "other_fresh"])
def test_opaque_storage_references_cannot_authorize_reuse_by_identity(origin):
    a, b, other = [tensor(name, [1], "int32", "shared.dyn") for name in ["a", "b", "other"]]
    pointer_type = ir.PtrType("int32", "shared.dyn")
    pointer = ir.Var("pointer", pointer_type)
    definitions, params = [allocate(a), allocate(b)], []
    if origin == "parameter":
        params.append(pointer)
    elif origin == "integer":
        address = ir.Var("address", "uint64")
        params.append(address)
        definitions.append(ir.Bind(pointer, ir.Call("tirx.reinterpret", [address], ty=pointer_type)))
    else:
        definitions.extend([allocate(other), ir.Bind(pointer, other.data)])
    wa, ra, wb, rb = _write(a), _read(a), _write(b), _read(b)
    opaque = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("opaque_storage_use"), pointer], ty="int32"))
    sequence = [wa, ra, opaque, wb, rb]
    edges = [(lhs, rhs, 0, True) for lhs, rhs in zip(sequence, sequence[1:])]
    value = _kernel(params, [*map(_unit, definitions), *map(_unit, sequence)])
    assert _can_alias(_aliases(value, edges), a, b) == (origin == "other_fresh")


@pytest.mark.parametrize("origin", ["parameter", "integer", "fresh"])
def test_external_onchip_tensor_views_remain_possible_hidden_lifetimes(origin):
    a, b, view = [tensor(name, [1], "int32", "shared.dyn") for name in ["a", "b", "view"]]
    definitions, params = [allocate(a), allocate(b)], []
    if origin == "fresh":
        definitions.append(allocate(view))
    else:
        address = ir.Var("address", ir.PtrType("int32", "shared.dyn") if origin == "parameter" else "uint64")
        params.append(address)
        pointer = address if origin == "parameter" else ir.Call("tirx.reinterpret", [address], ty=ir.PtrType("int32", "shared.dyn"))
        definitions.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])))
    wa, ra, wb, rb, external = _write(a), _read(a), _write(b), _read(b), _read(view)
    sequence = [wa, ra, external, wb, rb]
    value = _kernel(params, [*map(_unit, definitions), *map(_unit, sequence)])
    edges = [(lhs, rhs, 0, True) for lhs, rhs in zip(sequence, sequence[1:])]
    assert _can_alias(_aliases(value, edges), a, b) == (origin == "fresh")


@pytest.mark.parametrize("moving", [False, True])
def test_carried_definition_requires_physical_address_periodicity(moving):
    a, b = tensor("a", [32], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    view = tensor("view", [1], "int32", "shared.dyn")
    index = ir.Var("i", "int32")
    offset = ir.prim.Cast("int64", index * 4) if moving else integer(0, "int64")
    pointer = ir.Call("tirx.ptr_byte_offset", [a.data, offset], ty=a.data.ty)
    definition = ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))
    ra, wb, rb, wa = _read(view), _write(b), _read(b), _write(view)
    sequence = [ra, wb, rb, wa]
    value = _kernel([], [_unit(allocate(a)), _unit(allocate(b)), _loop(index, [_unit(definition), *map(_unit, sequence)])])
    edges = [(ra, wb, 0, True), (wb, rb, 0, True), (rb, wa, 0, True), (wa, ra, 1, True)]
    aliases = _aliases(value, edges, clock=index, periods={a: 1, b: 1}, recurrence=True)
    assert _can_alias(aliases, a, b) == (not moving)


@pytest.mark.parametrize("where", ["if", "else", "loop"])
def test_late_header_reads_extend_the_original_allocation_lifetime(where):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    wa, wb = _write(a, 2), _write(b)
    header = ir.TensorLoad(a, [integer(0)])
    read_b = _read(b)
    if where == "if":
        tail = ir.If(header > 0, [_unit(read_b)], None)
    elif where == "else":
        tail = ir.If(header > 0, [], [_unit(read_b)])
    else:
        tail = _loop(ir.Var("j", "int32"), [_unit(read_b)], end=header)
    value = _kernel([], [_unit(allocate(a)), _unit(allocate(b)), _unit(wa), _unit(wb), tail])
    aliases = _aliases(value, [(wa, wb, 0, True), (wb, header, 0, True), (header, read_b, 0, True)])
    assert not _can_alias(aliases, a, b)


def test_missing_inner_clock_does_not_turn_a_nested_loop_into_a_single_event():
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    index, flag = ir.Var("i", "int32"), ir.Var("flag", "bool")
    wa, ra, wb, rb = _write(a), _read(a), _write(b), _read(b)
    loop = _loop(index, list(map(_unit, [wa, ra, wb, rb])))
    value = _kernel([flag], [_unit(allocate(a)), _unit(allocate(b)), ir.If(flag, [loop], None)])
    aliases = _aliases(value, [(wa, ra, 0, True), (ra, wb, 0, True), (wb, rb, 0, True)])
    assert not _can_alias(aliases, a, b)


@pytest.mark.parametrize(
    "sites,edges",
    [
        (1, []),
        (3, [[0, 1, 0, 0], [1, 2, 0, 0]]),
        (3, [[0, 2, 0, 0], [0, 1, 4, 1], [1, 2, 3, 0]]),
        (2, [[0, 1, 0, 0], [1, 0, 0, 1]]),
        (3, [[0, 1, 2, 1], [1, 2, 3, 0], [2, 0, 4, 0], [0, 2, 8, 1]]),
        (3, [[0, 1, 2**62, 1], [1, 2, 2**62, 1], [0, 2, 2**63 - 1, 1]]),
    ],
)
def test_iteration_order_matches_bounded_walk_enumeration(sites, edges):
    # With nonnegative weights, a shortest qualifying walk has a simple path
    # in the (site, seen_completion) state space, hence at most 2*sites steps.
    expected = []
    for source in range(sites):
        distances = [None] * sites
        frontier = [(source, 0, False)]
        for _ in range(2 * sites):
            next_frontier = []
            for at, distance, completed in frontier:
                for start, end, delta, strong in edges:
                    if start != at or distance + delta >= 2**63:
                        continue
                    total, has_completion = distance + delta, completed or bool(strong)
                    if has_completion and (distances[end] is None or total < distances[end]):
                        distances[end] = total
                    next_frontier.append((end, total, has_completion))
            frontier = next_frontier
        expected.extend(distances)
    queries = [[source, destination] for source in range(sites) for destination in range(sites)]
    assert list(_api("AscendIterationOrder")(sites, edges, queries)) == expected


@pytest.mark.parametrize("edge", [[-1, 0, 0, 1], [0, 2, 0, 1], [0, 1, -1, 1]])
def test_iteration_order_rejects_invalid_sites_and_negative_distances(edge):
    with pytest.raises((ValueError, RuntimeError)):
        _api("AscendIterationOrder")(2, [edge], [[0, 1]])
