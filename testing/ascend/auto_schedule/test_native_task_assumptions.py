"""Task wrappers retain assumptions only within their native lexical scope."""

import pytest
import tilelang.ascend
from tilelang.ascend import transform
from testing.ascend._ir import attribute, calls, copy, kernel, seq, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm_tirx import ir, sym
from tvm_tirx.ir import prim


def _explicit_flags(flag, scheduled=True):
    statements = [
        ir.Evaluate(
            ir.Call(
                "tl.ascend_cross_core_" + action + "_flag",
                [ir.const(0), ir.StringImm("PIPE_FIX"), flag],
                ty=ir.PrimType("void"),
            )
        )
        for action in ("set", "wait")
    ]
    return seq(*[unit(stmt, core=2) if scheduled else stmt for stmt in statements])


def _flag_program(form):
    flag = ir.Var("flag", "int32")
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    temp = tensor((8, 16), "float32", name="temp", scope="shared.dyn")
    out = tensor((8, 16), "float32", name="out")
    condition = prim.And(flag >= 0, flag < 4)
    assumption = ir.Evaluate(ir.Call("prim.assume", [condition], ty=ir.PrimType("void")))
    explicit = _explicit_flags(flag)
    if form == "flat":
        explicit = unit(seq(assumption, _explicit_flags(flag, scheduled=False)), core=2)
    elif form == "task":
        explicit = seq(unit(assumption, core=2), explicit)
    elif form in ("tilelang_assume", "tl.assume_requires_runtime_check"):
        explicit = attribute(condition, form, "flag range", explicit)
    elif form == "sibling_scope":
        explicit = seq(attribute(condition, "tilelang_assume", "flag range", unit(ir.Evaluate(ir.const(0)), core=2)), explicit)
    elif form == "branch":
        branch = ir.Var("branch", "bool")
        explicit = seq(ir.If(branch, [unit(assumption, core=2)], None), explicit)
    body = seq(
        unit(copy(accum, temp, annotations={"dual_dst_ctl": 1}), core=2),
        explicit,
        unit(copy(temp, out), core=5),
    )
    return kernel(
        body,
        buffers=[accum, temp],
        params=[flag, out] + ([branch] if form == "branch" else []),
        allocation_stage=0,
    )


@pytest.mark.parametrize(
    "form",
    ["flat", "task", "tilelang_assume", "tl.assume_requires_runtime_check"],
)
def test_assumptions_bound_explicit_flag_reservations(form):
    tilelang.ascend.register_backend()
    after = transform.InsertSync()(_flag_program(form))
    generated = [
        call for action in ("set", "wait") for call in calls(after, "tl.ascend_cross_core_" + action + "_flag") if int(call.args[0]) == 4
    ]
    assert generated
    analyzer = sym.Analyzer()
    identifiers = [analyzer.simplify(call.args[2]) for call in generated]
    assert all(isinstance(identifier, ir.IntImm) for identifier in identifiers)
    assert all(4 <= int(identifier) % 16 < 16 for identifier in identifiers)


@pytest.mark.parametrize("form", ["unbounded", "sibling_scope", "branch"])
def test_assumption_scope_does_not_bound_unrelated_flag_uses(form):
    tilelang.ascend.register_backend()
    with pytest.raises((ValueError, RuntimeError), match="integer range cannot be bounded"):
        transform.InsertSync()(_flag_program(form))
