"""Original Ascend fixtures retain real dependencies when arguments may alias."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from testing.ascend._ir import attribute_fields, attributes, calls, disjoint_argument_fixture, schedule
from testing.ascend.auto_schedule import test_conflict_hints as hints
from testing.ascend.auto_schedule import test_flag_allocation as flags


def _graph(before):
    assert tirx.analysis.check_well_formed(before["main"])
    (root,) = [node for node in attributes(before, "tl.sblock") if attribute_fields(node)["value"]["name"] == "tilelang_root"]

    def api(name):
        return tvm_ffi.get_global_func("tl.testing." + name)

    snapshot = ir.save_json(before)
    plan = api("InspectAscendMultiBufferPlan")(root, [])
    versions = {row["storage"]: int(row["versions"]) for row in plan["infos"]}
    owners = {row["storage"]: [owner["loop"] for owner in row["owners"]] for row in plan["infos"]}
    graph = api("AnalyzeAscendDependencies")(root, versions, owners)
    assert snapshot == ir.save_json(before)
    return graph, plan


def _external_pairs(graph):
    return {
        (int(pair[0]), int(pair[1]), int(dependency["distance"]))
        for scope in graph["scopes"]
        for dependency in scope["dependencies"]
        if "storage" not in dependency
        for pair in dependency["pairs"]
    }


def _external_tasks(before, graph):
    a, c = before["main"].params
    reads = [int(row["id"]) for row in graph["tasks"] if any(region.source.same_as(a) for region in row["reads"])]
    writes = [int(row["id"]) for row in graph["tasks"] if any(region.source.same_as(c) for region in row["writes"])]
    assert reads and writes
    return reads, writes


@pytest.mark.parametrize(
    "make, versions, owner_count",
    [
        (flags._make_cross_iter_reuse_program, 2, 4),
        (flags._make_multi_owner_counter_channel_program, 8, 2),
        (flags._make_single_iteration_nested_loop_program, 3, 1),
        (flags._make_noneligible_single_iteration_nested_loop_program, 3, 1),
    ],
    ids=["sibling", "two-channels", "extent-one-owner", "extent-one-nonowner"],
)
def test_original_flag_fixtures_keep_argument_alias_dependencies(make, versions, owner_count):
    program = make()
    assert not any(key in (program.attrs or {}) for key in ("tir.noalias", "tirx.noalias"))
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    original = ir.save_json(before)
    graph, plan = _graph(before)
    (info,) = plan["infos"]
    assert int(info["versions"]) == versions
    assert len(info["owners"]) == owner_count
    reads, writes = _external_tasks(before, graph)
    pairs = _external_pairs(graph)
    # For C == A, each input must be read before the corresponding output is
    # written. With shifted C, a completed store can feed the next iteration's
    # load. These external hazards must not inherit UB's version distance.
    required = set()
    for write in writes:
        read = max(read for read in reads if read < write)
        required.update({(read, write, 0), (write, read, -1)})
    assert required <= pairs
    disjoint = disjoint_argument_fixture(before, ("A", "C"))
    disjoint_graph, disjoint_plan = _graph(disjoint)
    assert not required & _external_pairs(disjoint_graph)
    ir.assert_structural_equal(plan["infos"], disjoint_plan["infos"])
    assert original == ir.save_json(before)


@pytest.mark.parametrize("kind", ["same", "cross", "partial", "no-conflict"])
def test_original_local_hints_do_not_suppress_external_argument_hazards(kind):
    program = (
        hints._make_no_conflict_alias_program()
        if kind == "no-conflict"
        else hints._hint_program("loop", kind != "partial", kind == "cross", False)
    )
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    graph, _ = _graph(before)
    (read,), (write,) = _external_tasks(before, graph)
    required = {(read, write, 0), (write, read, -1)}
    assert required <= _external_pairs(graph)
    explicit, _ = _graph(disjoint_argument_fixture(before, ("A", "C")))
    # Exact local hints retain their requested distance. Partial hints do not
    # match a full access, and alias/UB no-conflict removes only that storage.
    pairs = _external_pairs(explicit)
    expected = {(read, write, -1), (write, read, -1)} if kind == "cross" else {(read, write, 0)} if kind == "same" else set()
    assert pairs == expected


def _sibling_copy_trace(before, graph):
    """Enumerate the real input's copy regions, independently of dependencies."""
    a, c = before["main"].params
    task_calls = [(call, int(row["id"])) for row in graph["tasks"] for call in calls(row["statement"], "tl.tileop.ascend_copy")]
    trace = []

    def integer(expression, values):
        analyzer = sym.Analyzer()
        constraints = sym.ConstraintContext()
        for variable, value in values.items():
            constraints.append(sym.BindingConstraint(variable, ir.IntImm(variable.ty.dtype, value)))
        analyzer.sync_constraint_context(constraints)
        result = analyzer.simplify(expression)
        assert isinstance(result, ir.IntImm)
        return int(result)

    def visit(node, values):
        if isinstance(node, ir.SeqStmt):
            for statement in node.seq:
                visit(statement, values)
        elif isinstance(node, ir.RegionStmt):
            visit(node.body, values)
        elif isinstance(node, ir.For):
            for index in range(integer(node.begin, values), integer(node.end, values), integer(node.step, values)):
                visit(node.body, {**values, node.var: index})
        elif isinstance(node, ir.Evaluate) and isinstance(node.value, ir.Call):
            call = node.value
            if not isinstance(call.op, ir.Op) or call.op.name != "tl.tileop.ascend_copy":
                return
            source, destination = call.args[:2]
            is_read = source.source.same_as(a)
            assert is_read or destination.source.same_as(c)
            access = source if is_read else destination
            assert len(access.region) == 1
            (region,) = access.region
            (task_id,) = [task_id for candidate, task_id in task_calls if candidate.same_as(call)]
            trace.append(("read" if is_read else "write", integer(region.begin, values), integer(region.extent, values), task_id))

    visit(before["main"].body, {})
    return trace


def test_sibling_shifted_argument_witness_requires_original_alias_edges():
    before = schedule(flags._make_cross_iter_reuse_program(), through="ResolveCore", auto_schedule=False)
    graph, _ = _graph(before)
    trace = _sibling_copy_trace(before, graph)
    assert [(role, offset, size) for role, offset, size, _ in trace] == [
        (role, offset, 64) for offset in range(0, 4096, 64) for role in ("read", "write")
    ]
    pairs = _external_pairs(graph)
    # Each adjacent tile shares physical bytes for C = A + 64. Check all 63
    # boundaries, including sibling-loop and outer-loop transitions.
    for previous_write, following_read in zip(trace[1::2], trace[2::2]):
        writer, reader = previous_write[3], following_read[3]
        assert previous_write[1] + 64 == following_read[1]
        assert (writer, reader, 0 if writer < reader else -1) in pairs

    def execute(prefetch, aliased):
        backing = np.arange(4160, dtype="float32")
        a = backing[:4096]
        c = backing[64:] if aliased else np.empty_like(a)
        assert np.shares_memory(a, c) == aliased
        if aliased:
            assert c.ctypes.data - a.ctypes.data == 64 * a.itemsize
        loaded = a[:64].copy()
        for index, (_, offset, size, _) in enumerate(trace[::2]):
            upcoming = a[offset + size : offset + 2 * size].copy() if prefetch and index < 63 else None
            if not prefetch:
                loaded = a[offset : offset + size].copy()
            c[offset : offset + size] = loaded
            loaded = upcoming
        return c.copy()

    sequential = execute(False, True)
    prefetched = execute(True, True)
    np.testing.assert_array_equal(sequential, np.tile(np.arange(64, dtype="float32"), 64))
    np.testing.assert_array_equal(prefetched, np.arange(4096, dtype="float32"))
    assert np.count_nonzero(sequential != prefetched) == 4032
    np.testing.assert_array_equal(execute(False, False), execute(True, False))
