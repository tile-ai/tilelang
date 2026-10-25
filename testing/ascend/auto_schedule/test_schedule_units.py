"""Schedule units contracts."""

import pytest
from tvm_tirx import ir
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang._compat import tirx
from testing.ascend._ir import root_scope, copy, kernel, nodes, seq, simd_scope, unseq
from testing.ascend.auto_schedule._scheduled_ir import _marker, unit
from testing.ascend.auto_schedule._task_utils import (
    _bind_target,
    _collect_schedule_units,
    _collect_task_metadata,
    _make_program,
    _materialize_schedule_units,
)


def _make_guarded_program():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2) as bx:
            temp = T.alloc_shared((16,), "float32")
            if bx == 0:
                T.copy(A, temp)
            else:
                T.copy(temp, B)

    return main


def _make_partially_staged_program():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(4, annotations={"enable_offset": True}):
                # Source order intentionally opposes logical stage order.
                with T.Stage(1), T.Task(latency=1, ii=1):
                    T.copy(temp, B[i * 16 : (i + 1) * 16])
                with T.Task(latency=1, ii=1):
                    T.copy(A[i * 16 : (i + 1) * 16], temp)

    return main


def _direct_schedule_unit_stages(stmt):
    statements = stmt.seq if isinstance(stmt, tirx.SeqStmt) else [stmt]
    return [
        int(statement.attrs["tl.schedule_unit"]["stage"])
        for statement in statements
        if isinstance(statement, ir.RegionStmt) and "tl.schedule_unit" in statement.attrs
    ]


def test_materialize_schedule_units_builds_trivial_guarded_schedule():
    mod = _materialize_schedule_units(_bind_target(_make_guarded_program()))
    schedule_units = _collect_schedule_units(mod)
    task_metadata = _collect_task_metadata(mod)

    assert schedule_units
    assert all(int(unit.attrs["tl.schedule_unit"]["stage"]) == -1 for unit in schedule_units)
    assert all(len(unit.attrs["tl.schedule_unit"]) == 1 for unit in schedule_units)
    conditions = [region.attrs["tl.schedule_condition"] for region in nodes(mod, ir.RegionStmt) if "tl.schedule_condition" in region.attrs]
    assert len(conditions) == 2
    (launch,) = [region for region in nodes(mod, ir.RegionStmt) if region.op.name == "tl.launch_grid"]
    for core in range(2):
        active = [
            tvm.sym.Analyzer().simplify(tirx.stmt_functor.substitute(condition, {launch.body_params[0]: core})) for condition in conditions
        ]
        assert sum(bool(condition) for condition in active) == 1
    assert task_metadata["latency"] == []
    assert task_metadata["ii"] == []


def test_materialize_schedule_units_normalizes_partial_manual_stages():
    mod = _materialize_schedule_units(_bind_target(_make_partially_staged_program()))
    loop_units = [unit for unit in _collect_schedule_units(mod) if isinstance(unseq(unit.body), tirx.For)]

    assert len(loop_units) == 1
    assert _direct_schedule_unit_stages(unseq(loop_units[0].body).body) == [1, 0]


def _scheduled_copies(*, cost=(1, 1), malformed=False, staged=False):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    producer = unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64]), core=0, cost=cost)
    consumer = unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64]), core=0, stage=1 if staged else 0, cost=cost)
    if malformed:
        body = _marker("tl.schedule_unit", {"stage": 0}, seq(producer.body, consumer.body))
    else:
        body = seq(consumer, producer) if staged else seq(producer, consumer)
    loop = tirx.For(
        i, 0, 4, 1, tirx.ForKind.DEFAULT, body, annotations={"enable_offset": staged, "num_stages": 2, "multi_buffer_eligible": [ub]}
    )
    return kernel(unit(loop, core=None), buffers=[ub], params=[a, c])


def test_auto_schedule_preserves_explicit_stage_dependencies():
    before = _scheduled_copies(staged=True)
    after = ascend_transform.AutoSchedule()(before)
    (loop_unit,) = [node for node in _collect_schedule_units(after) if isinstance(unseq(node.body), tirx.For)]
    # Stage order is explicitly requested by this input. Other scheduling
    # choices are deliberately left unasserted.
    assert _direct_schedule_unit_stages(unseq(loop_unit.body).body) == [0, 1]


@pytest.mark.parametrize(
    "case, message",
    [
        ("missing-cost", "EstimateLatency"),
        ("missing-units", "MaterializeScheduleUnits"),
        ("multiple-tasks", "exactly one outer T.Task/T.PerCoreTask"),
    ],
)
def test_auto_schedule_requires_its_input_contract(case, message):
    if case == "missing-units":
        before = _bind_target(_make_program())
    else:
        before = _scheduled_copies(cost=None if case == "missing-cost" else (1, 1), malformed=case == "multiple-tasks")
    with pytest.raises(ValueError if case == "missing-units" else RuntimeError, match=message):
        ascend_transform.AutoSchedule()(before)


@pytest.mark.parametrize("eligible", [False, True], ids=["ordinary-dependency", "owner-exclusion"])
@pytest.mark.parametrize("second_stage", [0, 1], ids=["same-stage", "offset-stage"])
def test_single_version_sibling_owners_preserve_dependency_kind(eligible, second_stage):
    a = tirx.decl_buffer((1,), "int32", name="A")
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    owners = []
    for index in range(2):
        i = tirx.Var(f"owner{index}", "int32")
        write = tirx.BufferStore(ub, index + 1, [0])
        if index == 0:
            write = simd_scope(write)
        # The shared MTE2 pipe fixes physical owner order. Without it, the
        # ordinary dependency can be satisfied by reversing modulo phases.
        body = seq(unit(copy(a, ub), core=0, stage=-1, cost=(1, 1)), unit(write, core=0, stage=-1, cost=(1, 1)))
        owner = tirx.For(i, 0, 1, 1, tirx.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [ub]} if eligible else {})
        owners.append(unit(owner, core=None, stage=second_stage if index else 0))
    outer = tirx.For(
        tirx.Var("outer", "int32"), 0, 4, 1, tirx.ForKind.DEFAULT, seq(*owners), annotations={"num_stages": 2, "enable_offset": True}
    )
    before = kernel(unit(outer, core=None, stage=-1), buffers=[ub], params=[a], annotations={"tl.buffer_versions_map": {ub: 1}})
    if not eligible and second_stage:
        with pytest.raises(RuntimeError, match="Manual schedule constraints are infeasible"):
            ascend_transform.AutoSchedule()(before)
        return
    after = ascend_transform.AutoSchedule()(before)
    (scheduled_outer,) = [loop for loop in nodes(after, tirx.For) if loop.var.same_as(outer.var)]
    assert _direct_schedule_unit_stages(scheduled_outer.body) == [0, second_stage]
    root = root_scope(after)
    versions = root.attrs.get("tl.buffer_versions_map", {})
    assert (ub in versions) == eligible
    if eligible:
        assert int(versions[ub]) == 1


@pytest.mark.parametrize(
    "fixed,size", [(False, 96), (True, 96), (False, 100)], ids=["inferred-fits", "fixed-fits", "inferred-exceeds-pool"]
)
def test_ss_versions_share_capacity_across_aivs(fixed, size):
    buffers = [tirx.decl_buffer((size,), "uint64", name=f"ss{sid}", scope="shared.ss") for sid in range(2)]
    out = tirx.decl_buffer((4, 2), "uint64", name="out")
    i = tirx.Var("i", "int32")
    tasks = []
    for sid, (buffer, core) in enumerate(zip(buffers, (1, 4))):
        tasks.append(unit(tirx.BufferStore(buffer, tirx.Cast("uint64", i), [0]), core=core, cost=(1, 1)))
        tasks.append(unit(tirx.BufferStore(out, buffer[0], [i, sid]), core=2, stage=2, cost=(1, 1)))
    loop = tirx.For(
        i,
        0,
        4,
        1,
        tirx.ForKind.DEFAULT,
        seq(*tasks),
        annotations={"num_stages": 3, "enable_offset": True, "multi_buffer_eligible": [buffer for buffer in buffers]},
    )
    annotations = {"tl.split_subcores": True}
    if fixed:
        annotations["tl.buffer_versions_map"] = {buffer: 2 for buffer in buffers}
    before = kernel(unit(loop, core=None), buffers=buffers, params=[out], annotations=annotations)
    if size == 100:
        with pytest.raises(RuntimeError, match="shared.ss|Manual schedule constraints are infeasible"):
            ascend_transform.AutoSchedule()(before)
        return
    after = ascend_transform.AutoSchedule()(before)
    root = root_scope(after)
    versions = root.attrs["tl.buffer_versions_map"]
    # Both AIVs consume the same 3 KiB pool. Two versions of each fit exactly.
    assert all(int(versions[buffer]) == 2 for buffer in buffers)


def test_sf_hazard_uses_the_data_ring_variable_in_the_solver():
    # The consumer is two stages later. A one-version SF recurrence cannot
    # satisfy this manual schedule; it must use the group's requested ring.
    data = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((32, 2), "uint16", name="sf", scope="shared.l0a.sf")
    source = tirx.decl_buffer((32, 128), data.dtype, name="source", scope="shared.l1")
    scale_source = tirx.decl_buffer((32, 2), sf.dtype, name="scale_source", scope="shared.l1")
    out = tirx.decl_buffer((4,), "uint16", name="out")
    i = tirx.Var("i", "int32")
    tasks = seq(
        unit(copy(source, data), core=2, cost=(1, 1)),
        unit(copy(scale_source, sf), core=2, cost=(1, 1)),
        unit(tirx.BufferStore(out, sf[0, 0], [i]), core=2, stage=2, cost=(1, 1)),
    )
    loop = tirx.For(
        i,
        0,
        4,
        1,
        tirx.ForKind.DEFAULT,
        tasks,
        annotations={"num_stages": 3, "enable_offset": True, "multi_buffer_eligible": [data, sf]},
    )
    before = kernel(
        unit(loop, core=None),
        buffers=[data, sf, source, scale_source],
        params=[out],
        annotations={
            "tl.l0_sf_bindings": {sf: data},
            "tl.buffer_versions_map": {sf: 3},
        },
    )
    after = ascend_transform.AutoSchedule()(before)
    root = root_scope(after)
    versions = root.attrs["tl.buffer_versions_map"]
    assert int(versions[data]) == int(versions[sf]) == 3
