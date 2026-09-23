"""Special-register dependencies participate in scheduling and core placement."""

import pytest
import tilelang.ascend.language as T
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, seq, task_core_masks
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
    writer = unit(tirx.Evaluate(T.ascend_set_copy_pad_value(-1.0, dtype="float32")), core=3)
    loop = unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, writer), core=None)
    reader = unit(copy(a, ub, annotations={"data_select": tirx.IntImm("int32", 1)}))
    before = kernel(seq(loop, reader), buffers=[ub], params=[a])
    after = transform.ResolveCore()(before)
    assert task_core_masks(after, "tl.ascend_set_copy_pad_value") == {1}


@pytest.mark.parametrize("operation", ["atomic", "loop-break"])
def test_shared_special_register_writer_covers_both_cores(operation):
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    writer = T.set_atomic("add", "float32") if operation == "atomic" else T.loop_break()
    exit_loop = tirx.Var("exit_loop", "bool")
    body = seq(
        unit(tirx.Evaluate(writer), core=3, guard=exit_loop if operation == "loop-break" else None),
        unit(copy(ub, out)),
        unit(copy(accum, out), core=2),
    )
    if operation == "loop-break":
        i = tirx.Var("i", "int32")
        body = unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body), core=None)
    before = kernel(body, buffers=[ub, accum], params=[out, exit_loop])
    after = transform.ResolveCore()(before)
    assert task_core_masks(after, writer.op.name) == {3}


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
        unit(tirx.Evaluate(setters[0]), core=3, cost=(1, 1)),
        unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, readers[0]), core=None),
        unit(tirx.Evaluate(setters[1]), core=3, cost=(1, 1)),
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
def test_cube_mode_writers_follow_nested_gemm_core(mode):
    before, setter = _cube_mode_program(mode)
    after = transform.ResolveCore()(before)
    assert len(calls(after, setter)) == 2
    assert task_core_masks(after, setter) == {2}
