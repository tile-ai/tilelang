"""Native launch selection is testable independently of the MPS device SDK."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.jit.adapter.torch.metal_launch import MetalLaunchPlan


@T.prim_func
def add_two(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
    with T.Kernel(2, threads=32) as bx:
        for i in T.Parallel(16):
            B[bx * 16 + i] = A[bx * 16 + i] + T.float32(2)


@pytest.fixture(scope="module")
def artifact():
    return tilelang.lower(
        add_two, target="metal", target_host={"kind": "llvm", "mcpu": "generic"}, enable_host_codegen=False, enable_device_compile=False
    )


@pytest.mark.parametrize("roundtrip", [False, True])
def test_real_pipeline_keeps_native_launch_dimensions(artifact, roundtrip):
    function, host, device = add_two, artifact.host_mod, artifact.device_mod
    if roundtrip:
        function, host, device = [ir.load_json(ir.save_json(value)) for value in (function, host, device)]
    before = ir.save_json(host)
    plan = MetalLaunchPlan(function, device, host)
    assert plan.kernel_name in artifact.kernel_source
    assert plan.argument_order == [0, 1]
    assert plan.block_info == [32, 1, 1]
    assert plan.grid_info == [2, 1, 1]
    assert plan.threads == [64, 1, 1]
    assert ir.save_json(host) == before


def _native_call_model(*, conditional=False, dynamic=False):
    # The public order and emitted symbol deliberately have no naming relation.
    b = ir.Var("same", tirx.TensorType((32,), "float32"))
    a = ir.Var("same", tirx.TensorType((32,), "float32"))
    unused = ir.Var("unused", tirx.TensorType((32,), "float32"))
    extent = ir.Var("not_static", "int32") if dynamic else ir.const(3)
    public = tirx.Function([b, unused, a, *([extent] if dynamic else [])], []).with_attr("global_symbol", "public")
    pa, pb = ir.Var("a", ir.PtrType("float32")), ir.Var("b", ir.PtrType("float32"))
    kernel = tirx.Function([pa, pb], []).with_attr("global_symbol", "chosen_shader").with_attr("calling_conv", 2)
    other = tirx.Function([pa, pb], []).with_attr("global_symbol", "unrelated").with_attr("calling_conv", 2)
    launch = ir.Evaluate(tirx.call_ffi_kernel("chosen_shader", a.data, b.data, extent, 16, launch_params=["blockIdx.x", "threadIdx.x"]))
    body = [ir.If(ir.const(True), [launch], [])] if conditional else [launch]
    entry = public.with_body(body)
    key = ir.GlobalVar("public")
    host = ir.IRModule({key: entry}).with_attr("tl.source_host_schema", 2)
    host = host.with_attr("tl.source_host_module", ir.IRModule({key: entry}))
    return public, ir.IRModule({"unrelated": other, "chosen_shader": kernel}), host


def test_device_symbol_and_argument_order_come_from_actual_call():
    public, device, host = _native_call_model()
    plan = MetalLaunchPlan(public, device, host)
    assert plan.kernel_name == "chosen_shader"
    assert plan.argument_order == [2, 0]
    assert plan.threads == [48, 1, 1]
    assert plan.block_info == [16, 1, 1]


@pytest.mark.parametrize("single_kernel", [False, True])
def test_device_arguments_need_a_native_call_site(single_kernel):
    public, device, _ = _native_call_model()
    if single_kernel:
        device = ir.IRModule({"chosen_shader": device["chosen_shader"]})
    with pytest.raises(ValueError, match="native host module"):
        MetalLaunchPlan(public, device)


@pytest.mark.parametrize("conversion", ["scalar-bitcast", "pointer-to-scalar", "pointer-to-pointer"])
def test_direct_forwarding_only_strips_same_address_casts(conversion):
    x = ir.Var("x", "float32") if conversion == "scalar-bitcast" else ir.Var("x", tirx.TensorType((32,), "float32"))
    value = x if conversion == "scalar-bitcast" else x.data
    dtype = ir.PrimType("int32") if conversion == "scalar-bitcast" else ir.PrimType("uint64")
    if conversion == "pointer-to-pointer":
        dtype = ir.PtrType("uint8")
    projected = ir.Call("tirx.reinterpret", [value], ty=dtype)
    public = tirx.Function([x], []).with_attr("global_symbol", "public")
    parameter = ir.Var("device_x", dtype)
    kernel = tirx.Function([parameter], []).with_attr("global_symbol", "shader").with_attr("calling_conv", 2)
    launch = ir.Evaluate(tirx.call_ffi_kernel("shader", projected, 1, launch_params=["threadIdx.x"]))
    entry = public.with_body([launch])
    key = ir.GlobalVar("public")
    host = ir.IRModule({key: entry}).with_attr("tl.source_host_schema", 2)
    host = host.with_attr("tl.source_host_module", ir.IRModule({key: entry}))
    device = ir.IRModule({"shader": kernel})
    for function in [public, kernel, entry]:
        assert tirx.analysis.check_well_formed(function)
    if conversion == "pointer-to-pointer":
        assert MetalLaunchPlan(public, device, host).argument_order == [0]
    else:
        with pytest.raises(ValueError, match="direct public kernel arguments"):
            MetalLaunchPlan(public, device, host)


@pytest.mark.parametrize("conditional,dynamic,message", [(True, False, "single direct"), (False, True, "static")])
def test_direct_adapter_does_not_silently_drop_host_semantics(conditional, dynamic, message):
    public, device, host = _native_call_model(conditional=conditional, dynamic=dynamic)
    with pytest.raises(ValueError, match=message):
        MetalLaunchPlan(public, device, host)


def test_unknown_launch_tag_is_rejected():
    public, device, host = _native_call_model()

    def change(call):
        if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_ffi_kernel":
            return ir.Call(call.op, call.args, attrs=tirx.op.CallFFIKernelAttr(["not_a_dimension", "threadIdx.x"]), ty=call.ty)
        return call

    host = tvm_ffi.structural_map(host, (ir.Call, change))
    with pytest.raises(ValueError, match="Unsupported Torch Metal launch field"):
        MetalLaunchPlan(public, device, host)


@tilelang.testing.requires_metal
def test_torch_mps_direct_launch_preserves_preallocated_output():
    import torch

    kernel = tilelang.compile(add_two, target="metal", execution_backend="torch", out_idx=[])
    a = torch.arange(32, dtype=torch.float32, device="mps")
    b = torch.empty_like(a)
    kernel(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(b.cpu(), a.cpu() + 2, rtol=0, atol=0)
