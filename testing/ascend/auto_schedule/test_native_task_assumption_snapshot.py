"""Only an immutable captured flag keeps its assumed range after a store."""

import pytest
import tilelang.ascend
from tilelang.ascend import transform
from testing.ascend._ir import attribute, calls, copy, kernel, seq, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_native_task_assumptions import _explicit_flags
from tvm_tirx import ir, sym
from tvm_tirx.ir import prim


def _memory_flag_program(form, snapshot):
    memory = tensor((1,), "int32", name="flag_memory")
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    temp = tensor((8, 16), "float32", name="temp", scope="shared.dyn")
    out = tensor((8, 16), "float32", name="out")
    value = ir.Var("captured_flag", "int32") if snapshot else ir.TensorLoad(memory, [0])
    capture = [unit(ir.Bind(value, ir.TensorLoad(memory, [0])), core=2)] if snapshot else []
    condition = prim.And(value >= 0, value < 4)
    assumption = ir.Evaluate(ir.Call("prim.assume", [condition], ty=ir.PrimType("void")))
    overwrite_and_use = seq(
        unit(ir.TensorStore(memory, [0], ir.const(100)), core=2),
        _explicit_flags(value),
    )
    if form == "task":
        body = seq(unit(assumption, core=2), overwrite_and_use)
    elif form == "flat":
        # A flat task still needs its native scheduling envelope; only the
        # assume/store/flag operations inside that task are unwrapped.
        body = unit(
            seq(
                assumption,
                ir.TensorStore(memory, [0], ir.const(100)),
                _explicit_flags(value, scheduled=False),
            ),
            core=2,
        )
    else:
        body = attribute(condition, "tilelang_assume", "entry flag range", overwrite_and_use)
    return kernel(
        seq(unit(copy(accum, temp, annotations={"dual_dst_ctl": 1}), core=2), *capture, body, unit(copy(temp, out), core=5)),
        buffers=[accum, temp],
        params=[memory, out],
        allocation_stage=0,
    )


@pytest.mark.parametrize("form", ["task", "attribute", "flat"])
def test_memory_flag_does_not_inherit_pre_store_range(form):
    tilelang.ascend.register_backend()
    # Source order establishes the old memory range, then changes the value.
    # A later load is 100, whose hardware flag slot is outside that old range.
    before = _memory_flag_program(form, snapshot=False)
    with pytest.raises((ValueError, RuntimeError), match="integer range cannot be bounded"):
        transform.InsertSync()(before)


@pytest.mark.parametrize("form", ["task", "attribute", "flat"])
def test_ssa_flag_snapshot_keeps_its_range_after_memory_overwrite(form):
    tilelang.ascend.register_backend()
    before = _memory_flag_program(form, snapshot=True)
    after = transform.InsertSync()(before)
    generated = [
        call for action in ("set", "wait") for call in calls(after, "tl.ascend_cross_core_" + action + "_flag") if int(call.args[0]) == 4
    ]
    assert generated
    analyzer = sym.Analyzer()
    identifiers = [analyzer.simplify(call.args[2]) for call in generated]
    assert all(isinstance(identifier, ir.IntImm) for identifier in identifiers)
    assert all(4 <= int(identifier) % 16 < 16 for identifier in identifiers)
    # The condition constrains the captured value, not the current contents.
    explicit = [call for call in calls(after, "tl.ascend_cross_core_set_flag") if int(call.args[0]) == 0]
    assert explicit and all(isinstance(call.args[2], ir.Var) for call in explicit)
