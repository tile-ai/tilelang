import tilelang
from tilelang import tvm


def test_same_device_cross_target_call_uses_callee_global_symbol():
    callee_gvar = tvm.ir.GlobalVar("callee_hint")
    callee = (
        tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0))
        .with_attr("target", tvm.target.Target("llvm"))
        .with_attr("global_symbol", "actual_callee")
    )
    caller = (
        tvm.tirx.PrimFunc(
            [],
            tvm.tirx.Evaluate(tvm.tirx.Call("int32", callee_gvar, [])),
        )
        .with_attr("target", tvm.target.Target({"kind": "llvm", "mcpu": "generic"}))
        .with_attr("global_symbol", "caller")
    )
    mod = tvm.IRModule({callee_gvar: callee, "caller": caller})

    after = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    call = after["caller"].body.value

    assert call.op.same_as(tvm.ir.Op.get("tirx.call_extern"))
    assert call.args[0].value == "actual_callee"


if __name__ == "__main__":
    tilelang.testing.main()
