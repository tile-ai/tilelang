from __future__ import annotations

import pytest
import tvm_ffi
import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_split_host_device():
    """SplitHostDevice divides a function at the "target" attribute"""

    @I.ir_module
    class Before:
        @T.function
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            T.evaluate(n)

    @I.ir_module
    class Expected:
        @T.function
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.call_ffi_kernel("main_kernel", n, launch_params=[])

        @T.function
        def main_kernel(n: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "main_kernel",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            T.evaluate(n)

    After = _tirx_passes.SplitHostDevice()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)

    assert hasattr(_tirx_passes, "SplitHostDevice")
    assert not hasattr(_tirx_passes, "AnnotateDeviceRegions")
    assert hasattr(_tirx_passes, "OutlineDeviceRegions")
    assert hasattr(_tirx_passes, "LowerDeviceKernelLaunch")


def test_separate_outlining_and_launch_lowering():
    @I.ir_module
    class Before:
        @T.function
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            T.evaluate(n)

    transform = _tirx_passes
    outlined = transform.OutlineDeviceRegions()(Before)
    assert isinstance(outlined["main"].body[0].value.op, tvm_tirx.ir.GlobalVar)
    lowered = transform.LowerDeviceKernelLaunch()(outlined)
    tvm_tirx.ir.assert_structural_equal(lowered, transform.SplitHostDevice()(Before))


@pytest.mark.parametrize("target_kind", ["cuda", "rocm", "metal"])
@pytest.mark.parametrize("legacy_zero", [False, True])
def test_device_launch_preserves_early_return(target_kind, legacy_zero):
    ir = tvm_tirx.ir
    tirx = tvm_tirx.tirx
    thread = ir.Var("thread", "int32")
    value = ir.const(0) if legacy_zero else ir.Tuple([])
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [thread],
        None,
        [ir.If(thread >= 16, ir.Return(value), None), ir.Evaluate(7)],
    )
    device = tirx.Function([], body, ret_type=value.ty)
    device = device.with_attr("target", tvm_tirx.target.Target(target_kind))
    symbol = ir.GlobalVar("device")
    host = tirx.Function([], ir.Evaluate(ir.Call(symbol, [], ty=value.ty)))
    host = host.with_attr("target", tvm_tirx.target.Target("c"))
    module = _tirx_passes.LowerDeviceKernelLaunch()(ir.IRModule({symbol: device, "host": host}))
    returned = []
    tvm_ffi.structural_visit(module["device"].body, [(ir.Return, lambda node, visitor: returned.append(node))])
    assert len(returned) == 1
    assert isinstance(returned[0].value, ir.Tuple) and len(returned[0].value) == 0
    assert module["device"].ret_type == ir.TupleType([])
    assert tirx.analysis.check_well_formed(module)


def test_cross_target_host_call_uses_exported_symbol():
    ir = tvm_tirx.ir
    tirx = tvm_tirx.tirx
    symbol = ir.GlobalVar("callee_hint")
    callee = tirx.Function([], ir.Evaluate(0))
    callee = callee.with_attr("target", tvm_tirx.target.Target("llvm"))
    callee = callee.with_attr("global_symbol", "actual_callee")
    caller = tirx.Function([], ir.Evaluate(ir.Call(symbol, [], ty=ir.TupleType([]))))
    caller = caller.with_attr("target", tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"}))
    module = _tirx_passes.LowerDeviceKernelLaunch()(ir.IRModule({symbol: callee, "caller": caller}))
    call = module["caller"].body.seq[0].value
    assert call.op.name == "tirx.call_extern"
    assert call.args[0].value == "actual_callee"
