"""CUDA boolean physical parameters follow actual native launch classification."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

import tilelang


def _module(kind="cuda", *, finalized=False, raw=False):
    output = ir.Var("output", ir.PtrType("int32", "global"))
    enabled = ir.Var("enabled", "bool")
    thread = ir.Var("thread", "int64")
    block = ir.Var("block", "int64")
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("blockIdx.x"), ir.const(1, "int64")],
        [block],
        {},
        [
            ir.RegionStmt(
                "tirx.launch_thread",
                [ir.StringImm("threadIdx.x"), ir.const(32, "int64")],
                [thread],
                {},
                [ir.Evaluate(prim.Cast("int32", enabled))],
            )
        ],
    )
    attributes = {"target": Target(kind), "global_symbol": "actual_kernel", "custom_flag": enabled}
    if finalized:
        attributes.update(calling_conv=2, **{"cuda.launch_fields": ["grid.x", "block.x"]})
    if raw:
        attributes["code_block_source"] = 'extern "C" __global__ void actual_kernel(int*, bool) {}'
    device = tirx.Function([output, enabled], [body], ret_type=ir.PrimType("void"), attrs=ir.DictAttrs(attributes))
    global_var = ir.GlobalVar("kernel_hint")
    host_output = ir.Var("host_output", output.ty)
    host_enabled = ir.Var("enabled", "bool")
    if finalized:
        call = ir.Call(
            "tirx.call_ffi_kernel",
            [ir.StringImm("actual_kernel"), host_output, host_enabled, ir.const(1), ir.const(32)],
            attrs=tirx.CallFFIKernelAttr(launch_fields=["grid.x", "block.x"], num_kernel_args=2),
        )
    else:
        call = ir.Call(global_var, [host_output, host_enabled], ty="void")
    host = tirx.Function(
        [host_output, host_enabled],
        [ir.Evaluate(call)],
        attrs=ir.DictAttrs({"target": Target(kind, host="c"), "global_symbol": "main"}),
    )
    return ir.IRModule({global_var: device, "main": host}), global_var


def _launches(function):
    found = []
    tvm_ffi.structural_walk(
        function.body,
        (ir.Call, lambda node: found.append(node) if getattr(node.op, "name", "") == "tirx.call_ffi_kernel" else None),
    )
    return found


@pytest.mark.parametrize("finalized", [False, True])
def test_global_and_finalized_cuda_calls_share_the_same_bool_storage(finalized):
    module, global_var = _module(finalized=finalized)
    before = ir.save_json(module)
    lower = tilelang.transform.LowerDeviceKernelLaunch()
    result = lower(module)
    device = result[global_var]
    assert device.params[1].ty == ir.PrimType("int32")
    assert result["main"].params[1].ty == ir.PrimType("bool")
    (call,) = _launches(result["main"])
    assert isinstance(call.args[2], prim.Cast)
    assert call.args[2].ty == ir.PrimType("int32")
    assert call.args[2].value.same_as(result["main"].params[1])
    custom_flag = device.attrs["custom_flag"]
    assert isinstance(custom_flag, prim.Cast) and custom_flag.ty == ir.PrimType("bool")
    assert custom_flag.value.same_as(device.params[1])
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])
    assert not tirx.analysis.undefined_vars(device)
    ir.assert_structural_equal(lower(result), result)
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    assert tirx.analysis.check_well_formed(tirx.transform.ConvertSSA()(result))


@pytest.mark.parametrize("kind,raw", [("hip", False), ("cuda", True)])
def test_non_cuda_and_raw_source_parameters_keep_their_external_type(kind, raw):
    module, global_var = _module(kind, raw=raw)
    result = tilelang.transform.LowerDeviceKernelLaunch()(module)
    assert result[global_var].params[1].ty == ir.PrimType("bool")
    (call,) = _launches(result["main"])
    assert call.args[2].ty == ir.PrimType("bool")


def test_cuda_subroutine_boolean_interface_is_unchanged():
    flag = ir.Var("flag", "bool")
    target = Target("cuda")
    symbol = ir.GlobalVar("helper")
    helper = tirx.Function(
        [flag],
        [ir.Return(flag)],
        ret_type=ir.PrimType("bool"),
        attrs=ir.DictAttrs({"target": target, "global_symbol": "helper", "tirx.is_global_func": True}),
    )
    host_flag = ir.Var("flag", "bool")
    caller = tirx.Function(
        [host_flag],
        [ir.Evaluate(ir.Call(symbol, [host_flag], ty="bool"))],
        attrs=ir.DictAttrs({"target": target, "global_symbol": "caller"}),
    )
    module = ir.IRModule({symbol: helper, "caller": caller})
    result = tilelang.transform.LowerDeviceKernelLaunch()(module)
    ir.assert_structural_equal(result, module)


class _IndexOnly:
    def __index__(self):
        return 2


class _IntOnly:
    def __int__(self):
        return 2


def _bind_cutedsl_boolean(value):
    from tilelang.jit.adapter.abi import SourceABIPlan
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    flag = ir.Var("flag", "bool")
    function = tirx.Function([flag], [ir.Evaluate(0)])
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    arguments = [value]
    adapter._bind_source_arguments(SourceABIPlan(function), arguments)
    return arguments[0]


@pytest.mark.parametrize(
    "value,expected",
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        (2, False),
        (3, True),
        (-1, True),
        (-2, False),
        (255, True),
        (256, False),
        (2**31, False),
        (2**32 + 1, True),
        (2**63 - 1, True),
        (np.int64(2), False),
        (np.uint64(3), True),
    ],
)
def test_cutedsl_boolean_binder_preserves_old_integer_carriers(value, expected):
    assert _bind_cutedsl_boolean(value) is expected


@pytest.mark.parametrize("value", [0.5, "true", None, _IndexOnly(), _IntOnly()])
def test_cutedsl_boolean_binder_rejects_non_integral_objects(value):
    with pytest.raises(TypeError, match="requires an integer"):
        _bind_cutedsl_boolean(value)


@pytest.mark.parametrize("value", [2**63, 2**64 - 1, 2**65, -(2**63) - 1])
def test_cutedsl_boolean_binder_rejects_values_outside_old_int64_carrier(value):
    with pytest.raises(OverflowError, match="int64 carrier"):
        _bind_cutedsl_boolean(value)
