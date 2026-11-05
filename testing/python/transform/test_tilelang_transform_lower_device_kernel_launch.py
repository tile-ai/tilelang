import tilelang
import pytest
import tvm_tirx as tvm
from tvm_tirx import ir, tirx


def test_same_device_cross_target_call_uses_callee_global_symbol():
    callee_gvar = tvm.ir.GlobalVar("callee_hint")
    callee = (
        tirx.Function([], [ir.Return(ir.const(0, "int32"))], ret_type=ir.PrimType("int32"))
        .with_attr("target", tvm.target.Target("llvm"))
        .with_attr("global_symbol", "actual_callee")
    )
    caller = (
        tirx.Function(
            [],
            [ir.Evaluate(ir.Call(callee_gvar, [], ty="int32"))],
        )
        .with_attr("target", tvm.target.Target({"kind": "llvm", "mcpu": "generic"}))
        .with_attr("global_symbol", "caller")
    )
    mod = tvm.IRModule({callee_gvar: callee, "caller": caller})

    after = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    call = after["caller"].body[0].value

    assert call.op.same_as(tvm.ir.Op.get("tirx.call_extern"))
    assert call.args[0].value == "actual_callee"


@pytest.mark.parametrize("nested", [False, True])
def test_existing_extern_name_is_preserved_when_nested_argument_is_lowered(nested):
    symbol = ir.GlobalVar("callee_hint")
    callee = tirx.Function(
        [],
        [ir.Return(ir.const(7, "int32"))],
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs({"global_symbol": "actual_callee", "target": tvm.target.Target("llvm")}),
    )
    argument = ir.Call(symbol, [], ty="int32") if nested else ir.const(2, "int32")
    explicit = ir.Call("tirx.call_extern", [ir.StringImm("callee_hint"), argument], ty="int32")
    caller = tirx.Function(
        [],
        [ir.Evaluate(explicit), ir.Evaluate(ir.Call(symbol, [], ty="int32"))],
        attrs=ir.DictAttrs({"global_symbol": "caller", "target": tvm.target.Target({"kind": "llvm", "mcpu": "generic"})}),
    )
    module = tvm.IRModule({symbol: callee, "caller": caller}, attrs=ir.DictAttrs({"test_marker": 37}))
    original = ir.save_json(module)
    lowered = tilelang.transform.LowerDeviceKernelLaunch()(module)
    explicit_result = lowered["caller"].body[0].value
    assert explicit_result.args[0].value == "callee_hint"
    if nested:
        assert explicit_result.args[1].args[0].value == "actual_callee"
    assert lowered["caller"].body[1].value.args[0].value == "actual_callee"
    assert lowered[symbol].attrs["global_symbol"] == "actual_callee"
    assert bool(lowered[symbol].attrs["tirx.is_global_func"])
    assert int(lowered.attrs["test_marker"]) == 37
    assert ir.save_json(module) == original
    ir.assert_structural_equal(tilelang.transform.LowerDeviceKernelLaunch()(lowered), lowered)


def test_same_target_globalvar_edge_is_preserved():
    symbol = ir.GlobalVar("callee_hint")
    target = tvm.target.Target("llvm")
    callee = tirx.Function(
        [],
        [ir.Return(ir.const(1, "int32"))],
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs({"target": target, "global_symbol": "actual_callee"}),
    )
    caller = tirx.Function(
        [],
        [ir.Evaluate(ir.Call(symbol, [], ty="int32"))],
        attrs=ir.DictAttrs({"target": target, "global_symbol": "caller"}),
    )
    module = tvm.IRModule({symbol: callee, "caller": caller})
    after = tilelang.transform.LowerDeviceKernelLaunch()(module)
    assert after["caller"].body[0].value.op.same_as(symbol)


if __name__ == "__main__":
    tilelang.testing.main()
