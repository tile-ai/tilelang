"""Special-register dependencies participate in scheduling and core placement."""

import inspect
from unittest.mock import patch

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, nodes, seq, task_core_masks
from testing.ascend.auto_schedule._scheduled_ir import unit


def test_nested_pipe_wait_orders_the_following_pipe_user():
    a = tirx.decl_buffer((64,), "float32", name="A")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    wait = tirx.Evaluate(T.ascend_sync_inter_wait("PIPE_MTE2", 3))
    loop = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, unit(wait, core=0, cost=(1, 1)))
    before = kernel(seq(unit(loop, core=None), unit(copy(a, ub), core=0, cost=(1, 1))), buffers=[ub], params=[a])
    after = transform.AutoSchedule()(before)
    # This order is a dependency on the pipe register, not a scheduler tie-break.
    operations = [call.op.name for call in calls(after, "tl.ascend_cross_core_wait_flag") + calls(after, "tl.tileop.ascend_copy")]
    assert len(operations) == 2
    children = after["main"].body.block.body.seq
    assert calls(children[0], "tl.ascend_cross_core_wait_flag")
    assert calls(children[1], "tl.tileop.ascend_copy")


def test_nested_pad_writer_follows_the_reader_core():
    a = tirx.decl_buffer((64,), "float32", name="A")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    writer = unit(tirx.Evaluate(T.ascend_set_copy_pad_value(-1.0, dtype="float32")), core=7)
    loop = unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, writer), core=None)
    reader = unit(copy(a, ub, annotations={"data_select": tirx.IntImm("int32", 1)}))
    before = kernel(seq(loop, reader), buffers=[ub], params=[a])
    after = transform.ResolveCore()(before)
    assert task_core_masks(after, "tl.ascend_set_copy_pad_value") == {5}


@pytest.mark.parametrize("operation", ["padding", "atomic"])
def test_projected_special_register_writers_keep_their_subcore(operation):
    a = tirx.decl_buffer((16,), "float32", name="A")
    ub = tirx.decl_buffer((16,), "float32", name="ub", scope="shared.dyn")
    sid = tirx.Var("sid", "int32")
    setters = (
        [T.ascend_set_copy_pad_value(value, dtype="float32") for value in (1.0, 2.0)]
        if operation == "padding"
        else [T.set_atomic("add", "float32"), T.set_atomic_none()]
    )
    reader = copy(a, ub, annotations={"data_select": tirx.IntImm("int32", 1)}) if operation == "padding" else copy(ub, a)
    body = seq(*(unit(tirx.Evaluate(setter), core=0, guard=sid == index) for index, setter in enumerate(setters)), unit(reader, core=0))
    before = kernel(body, buffers=[ub], params=[a])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    assigned = transform.AssignCore()(before)
    for mod, masks in ((assigned, {0: 3, 1: 6}), (transform.ResolveCore()(assigned), {0: 1, 1: 4})):
        writers = [
            task
            for task in nodes(mod, tirx.AttrStmt)
            if task.attr_key == "tl.ascend_task" and any(calls(task, setter.op.name) for setter in setters)
        ]
        assert len(writers) == 2
        assert {int(task.node["subcore"]): int(task.node["core_mask"]) for task in writers} == masks


@pytest.mark.parametrize("operation", ["atomic", "loop-break"])
def test_shared_special_register_writer_covers_both_cores(operation):
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    writer = T.set_atomic("add", "float32") if operation == "atomic" else T.loop_break()
    exit_loop = tirx.Var("exit_loop", "bool")
    body = seq(
        unit(tirx.Evaluate(writer), core=7, guard=exit_loop if operation == "loop-break" else None),
        unit(copy(ub, out)),
        unit(copy(accum, out), core=2),
    )
    if operation == "loop-break":
        i = tirx.Var("i", "int32")
        body = unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body), core=None)
    before = kernel(body, buffers=[ub, accum], params=[out, exit_loop])
    after = transform.ResolveCore()(before)
    assert task_core_masks(after, writer.op.name) == {7}


def _cube_mode_program(mode):
    a = tirx.decl_buffer((16, 16), "float32", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((16, 16), "float32", name="b", scope="shared.l0b")
    outputs = [tirx.decl_buffer((16, 16), "float32", name=f"c{i}", scope="shared.l0c") for i in range(2)]
    setters = (
        [T.set_hf32_mode("nearest_even"), T.set_hf32_mode(None)]
        if mode == "hf32"
        else [T.set_mmad_direction("n"), T.set_mmad_direction("m")]
    )
    readers = [unit(tirx.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)), core=2, cost=(64, 32)) for c in outputs]
    i = tirx.Var("i", "int32")
    # Explicit scheduled IR: operand producers are outside this pass's contract.
    body = seq(
        unit(tirx.Evaluate(setters[0]), core=7, cost=(1, 1)),
        unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, readers[0]), core=None),
        unit(tirx.Evaluate(setters[1]), core=7, cost=(1, 1)),
        readers[1],
    )
    return kernel(body, buffers=[a, b, *outputs]), setters[0].op.name


@pytest.mark.parametrize("mode", ["hf32", "mmad-direction"])
def test_cube_mode_changes_order_nested_and_following_gemms(mode):
    before, setter = _cube_mode_program(mode)
    after = transform.AutoSchedule()(before)
    children = after["main"].body.block.body.seq
    assert len(children) == 4
    assert calls(children[0], setter)
    assert calls(children[1], "tl.tileop.gemm")
    assert calls(children[2], setter)
    assert calls(children[3], "tl.tileop.gemm")


@pytest.mark.parametrize("mode", ["hf32", "mmad-direction"])
@pytest.mark.parametrize(
    "enable_offset, start_times, ii, order, stages",
    [
        pytest.param(True, [0, 50, 151, 152], 152, [3, 0, 1, 2], [1, 0, 0, 0], id="cross-iteration-tie"),
        pytest.param(True, [0, 50, 50, 82], 153, [0, 1, 2, 3], [0, 0, 0, 0], id="same-iteration-tie"),
        pytest.param(False, [0, 50, 151, 152], 153, [0, 1, 2, 3], [0, 0, 0, 0], id="offset-disabled"),
    ],
)
def test_cube_mode_schedule_preserves_iteration_order(mode, enable_offset, start_times, ii, order, stages):
    a = tirx.decl_buffer((16, 16), "float32", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((16, 16), "float32", name="b", scope="shared.l0b")
    outputs = [tirx.decl_buffer((16, 16), "float32", name=f"c{i}", scope="shared.l0c") for i in range(2)]
    setters = (
        [T.set_hf32_mode("nearest_even"), T.set_hf32_mode(None)]
        if mode == "hf32"
        else [T.set_mmad_direction("n"), T.set_mmad_direction("m")]
    )
    operations = [
        tirx.Evaluate(setters[0]),
        tirx.Evaluate(T.gemm(a, b, outputs[0], transpose_B=True, clear_accum=True)),
        tirx.Evaluate(setters[1]),
        tirx.Evaluate(T.gemm(a, b, outputs[1], transpose_B=True, clear_accum=True)),
    ]
    # Explicit input to AutoSchedule; operand producers are outside this pass.
    body = seq(*(unit(op, core=0, stage=-1, cost=(1, 1) if index % 2 == 0 else (64, 32)) for index, op in enumerate(operations)))
    loop = tirx.For(tirx.Var("i", "int32"), 0, 3, tirx.ForKind.SERIAL, body, annotations={"num_stages": 2, "enable_offset": enable_offset})
    before = kernel(unit(loop, core=None, stage=-1), buffers=[a, b, *outputs])
    # Fix the solver output to exercise result reconstruction independently of
    # Z3's choice among valid schedules. At II=152, the stage-1 GEMM_c1 and
    # stage-0 setter both have folded issue time 0: GEMM_c1(0) must precede
    # setter(1) at absolute time 152. No buffers are multi-buffer eligible,
    # so the solver's buffer_versions result is empty.
    with patch.object(z3_scheduler, "z3_schedule_loop_python", return_value=(start_times, [], ii)):
        after = transform.AutoSchedule()(before)
    children = after["main"].body.block.body.body.body.seq
    assert [int(child.node["stage"]) for child in children] == stages
    for child, index in zip(children, order):
        tvm.ir.assert_structural_equal(child.body.body, operations[index])


@pytest.mark.parametrize("stages", [(0, 0), (0, 1), (6, 10), (1, 0)], ids=["same", "ascending", "nonadjacent", "descending"])
def test_manual_scalar_issue_order_survives_stage_ties(stages):
    operations = [tirx.Evaluate(T.set_hf32_mode("nearest_even")), tirx.Evaluate(T.set_mmad_direction("n"))]
    body = seq(*(unit(op, core=0, stage=stage, cost=(1, 1)) for stage, op in zip(stages, operations)))
    loop = tirx.For(tirx.Var("i", "int32"), 0, 3, tirx.ForKind.SERIAL, body, annotations={"enable_offset": True})
    before = kernel(unit(loop, core=None, stage=-1))
    # These independent setters share PIPE_Scalar, which has no resource
    # exclusion. Ascending stages need distinct phases so the reconstruction
    # tie-break cannot reverse manual source order; other stage orders may tie.
    after = transform.AutoSchedule()(before)
    children = after["main"].body.block.body.body.body.seq
    assert [int(child.node["stage"]) for child in children] == list(stages)
    for child, op in zip(children, operations):
        tvm.ir.assert_structural_equal(child.body.body, op)


@pytest.mark.parametrize("cores", [(1, 4), (1, 2), (2, 4)], ids=["aiv0-aiv1", "aiv0-aic", "aic-aiv1"])
@pytest.mark.parametrize("in_loop", [False, True], ids=["sequence", "loop"])
def test_manual_pipe_order_does_not_order_distinct_physical_cores(cores, in_loop):
    sources = [tirx.decl_buffer((64,), "float32", name=f"A{i}") for i in range(2)]
    buffers = [
        tirx.decl_buffer((64,), "float32", name=f"tile{i}", scope="shared.l1" if core == 2 else "shared.dyn")
        for i, core in enumerate(cores)
    ]
    stages = (0, 1) if in_loop else (0, 0)
    body = seq(*(unit(copy(source, buffer), core=core, stage=stage, cost=(1, 1))
                 for source, buffer, core, stage in zip(sources, buffers, cores, stages)))
    if in_loop:
        loop = tirx.For(tirx.Var("i", "int32"), 0, 3, tirx.ForKind.SERIAL, body, annotations={"enable_offset": True})
        body = unit(loop, core=None, stage=-1)
    name = "z3_schedule_loop_python" if in_loop else "z3_schedule_seq_python"
    original = getattr(z3_scheduler, name)
    inputs = []

    def capture(*args, **kwargs):
        inputs.append(dict(inspect.signature(original).bind(*args, **kwargs).arguments))
        return original(*args, **kwargs)

    with patch.object(z3_scheduler, name, side_effect=capture):
        transform.AutoSchedule()(kernel(body, buffers=buffers, params=sources))
    (actual,) = inputs
    # Equal pipe names on separate cores do not impose manual issue order.
    # Shared HBM resources may independently require resource exclusion.
    assert actual["pipe_order_deps"] == []


@pytest.mark.parametrize("mode", ["hf32", "mmad-direction"])
def test_cube_mode_writers_follow_nested_gemm_core(mode):
    before, setter = _cube_mode_program(mode)
    after = transform.ResolveCore()(before)
    assert len(calls(after, setter)) == 2
    assert task_core_masks(after, setter) == {2}
