"""Guard-form controls for the flag collector's mutable-memory contract."""

import pytest

import tilelang.ascend
from tilelang.ascend import transform
from testing.ascend._ir import attribute, calls, copy, kernel, seq, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_native_task_assumptions import _explicit_flags
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir import prim


def _guard_program(kind, mutable):
    memory = tensor((1,), "int32", name="guard_memory")
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    temp = tensor((8, 16), "float32", name="temp", scope="shared.dyn")
    output = tensor((8, 16), "float32", name="output")
    flag = ir.TensorLoad(memory, [0]) if mutable else ir.Var("immutable_flag", "int32")
    condition = prim.And(flag >= 0, flag < 4)
    body = seq(unit(ir.TensorStore(memory, [0], ir.const(100)), core=2), _explicit_flags(flag))
    if kind == "if":
        guarded = ir.If(condition, [body], None)
    else:
        guarded = attribute(condition, "tl.assume_requires_runtime_check", "entry range", body)
    module = kernel(
        seq(
            unit(copy(accum, temp, annotations={"dual_dst_ctl": 1}), core=2),
            guarded,
            unit(copy(temp, output), core=5),
        ),
        buffers=[accum, temp],
        params=[memory, output] + ([] if mutable else [flag]),
        allocation_stage=0,
    )
    assert tirx.analysis.check_well_formed(module["main"])
    return module


@pytest.mark.parametrize("kind", ["if", "runtime_attribute"])
def test_guard_memory_value_is_not_replayed_after_write(kind):
    tilelang.ascend.register_backend()
    # The guard describes the old loaded value. The later flag load is 100;
    # assigning a fresh generated flag from [0, 3] could collide with 100 % 16.
    with pytest.raises((ValueError, RuntimeError), match="integer range cannot be bounded"):
        transform.InsertSync()(_guard_program(kind, mutable=True))


def test_immutable_parameter_guard_survives_other_memory_write():
    tilelang.ascend.register_backend()
    after = transform.InsertSync()(_guard_program("if", mutable=False))
    generated = [
        call for action in ("set", "wait") for call in calls(after, "tl.ascend_cross_core_" + action + "_flag") if int(call.args[0]) == 4
    ]
    assert generated
    analyzer = sym.Analyzer()
    identifiers = [analyzer.simplify(call.args[2]) for call in generated]
    assert all(isinstance(identifier, ir.IntImm) for identifier in identifiers)
    assert all(4 <= int(identifier) % 16 < 16 for identifier in identifiers)
