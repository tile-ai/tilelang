"""Schedule units contracts."""

import pytest
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
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
        int(statement.node["stage"])
        for statement in statements
        if isinstance(statement, tirx.AttrStmt) and statement.attr_key == "tl.schedule_unit"
    ]


def test_materialize_schedule_units_builds_trivial_guarded_schedule():
    mod = _materialize_schedule_units(_bind_target(_make_guarded_program()))
    schedule_units = _collect_schedule_units(mod)
    task_metadata = _collect_task_metadata(mod)

    assert schedule_units
    assert all(int(unit.node["stage"]) == -1 for unit in schedule_units)
    assert all(len(unit.node) == 1 for unit in schedule_units)
    assert sum(isinstance(unit.body, tirx.IfThenElse) for unit in schedule_units) >= 2
    assert task_metadata["latency"] == []
    assert task_metadata["ii"] == []


def test_materialize_schedule_units_normalizes_partial_manual_stages():
    mod = _materialize_schedule_units(_bind_target(_make_partially_staged_program()))
    loop_units = [unit for unit in _collect_schedule_units(mod) if isinstance(unit.body, tirx.For)]

    assert len(loop_units) == 1
    assert _direct_schedule_unit_stages(loop_units[0].body.body) == [1, 0]


def _scheduled_copies(*, cost=(1, 1), malformed=False, staged=False):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    producer = unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64]), core=0, cost=cost)
    consumer = unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64]), core=0, stage=1 if staged else 0, cost=cost)
    if malformed:
        body = tirx.AttrStmt({"stage": 0}, "tl.schedule_unit", 1, seq(producer.body, consumer.body))
    else:
        body = seq(consumer, producer) if staged else seq(producer, consumer)
    loop = tirx.For(
        i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"enable_offset": staged, "num_stages": 2, "multi_buffer_eligible": [ub.data]}
    )
    return kernel(unit(loop, core=None), buffers=[ub], params=[a, c])


def test_auto_schedule_preserves_explicit_stage_dependencies():
    before = _scheduled_copies(staged=True)
    after = ascend_transform.AutoSchedule()(before)
    (loop_unit,) = [node for node in _collect_schedule_units(after) if isinstance(node.body, tirx.For)]
    # Stage order is explicitly requested by this input. Other scheduling
    # choices are deliberately left unasserted.
    assert _direct_schedule_unit_stages(loop_unit.body.body) == [0, 1]


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
    with pytest.raises(tvm.error.InternalError, match=message):
        ascend_transform.AutoSchedule()(before)
