"""Native source-launch consumers preserve call operands and lexical control flow.

HIP uses the registered TileLang DSL pipeline. Ascend exercises the native
host/device ABI directly while its complete scheduling pipeline is integrated.
These tests generate source; matching SDK and device execution are separate.
"""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.jit.adapter.source_host import SourceHostProgram
from tilelang.jit.adapter.wrapper import (
    TLAscendSourceWrapper,
    TLCUDASourceWrapper,
    TLHIPSourceWrapper,
    TLWrapper,
)
from test_native_backend_pipeline import TARGETS, elementwise, shared_copy


@pytest.mark.parametrize("case", [elementwise, shared_copy])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_public_pipeline_to_native_source_wrapper(case, roundtrip, tmp_path):
    function = case()
    target = Target(TARGETS["hip"])
    artifact = tilelang.lower(function, target=target, target_host={"kind": "llvm", "mcpu": "generic"})
    host, device = artifact.host_mod, artifact.device_mod
    if roundtrip:
        function = ir.load_json(ir.save_json(function))
        host, device = ir.load_json(ir.save_json(host)), ir.load_json(ir.save_json(device))
    wrapper = TLWrapper(target)
    wrapper.assign_optimized_module(ir.IRModule({"main": function}))
    wrapper.assign_host_module(host)
    wrapper.assign_device_module(device)
    source = wrapper.wrap(artifact.kernel_source)
    assert "tl_source_hip_launch((const void*)main_kernel" in source
    assert "hipLaunchKernel" in source and "hipGetLastError" in source
    assert "cudaLaunch" not in source and "launch_arg" in source
    (tmp_path / "kernel.cpp").write_text(source)
    (tmp_path / "host.json").write_text(ir.save_json(host))


@pytest.mark.parametrize("kind", ["cuda", "hip"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_direct_source_wrapper_uses_native_host_launches(kind, roundtrip):
    function = elementwise()
    target = Target(TARGETS[kind])
    artifact = tilelang.lower(function, target=target, target_host={"kind": "llvm", "mcpu": "generic"})
    host, device = artifact.host_mod, artifact.device_mod
    if roundtrip:
        function, host, device = (ir.load_json(ir.save_json(value)) for value in (function, host, device))
    before = ir.save_json(host)
    wrapper_class = TLCUDASourceWrapper if kind == "cuda" else TLHIPSourceWrapper
    wrapper = wrapper_class(
        ir.IRModule({"main": function}),
        artifact.kernel_source,
        target,
        device_mod=device,
        host_mod=host,
    )
    assert wrapper.lib_code.startswith(artifact.kernel_source)
    assert f"tl_source_{kind}_launch((const void*)main_kernel" in wrapper.lib_code
    assert ir.save_json(host) == before


def native_launch_case(*, tags=None, launch_values=None, dtype="float32"):
    n, step = ir.Var("n", "int64"), ir.Var("step", "int32")
    a = ir.Var("same_name", tirx.TensorType([n], dtype, layout="default", data_alignment=1))
    b = ir.Var("same_name", tirx.TensorType([n], dtype, layout="default", data_alignment=1))
    parameters = [ir.Var("lhs", ir.PtrType(dtype, "global")), ir.Var("rhs", ir.PtrType(dtype, "global")), ir.Var("iteration", "int32")]
    device = tirx.Function(
        parameters,
        [],
        ir.TupleType([]),
        ir.DictAttrs(
            {
                "global_symbol": "compute_kernel",
                "calling_conv": 2,
                "target": Target("ascend"),
            }
        ),
    )
    tags = tags if tags is not None else ["blockIdx.x", "tirx.use_dyn_shared_memory"]
    launch_values = launch_values if launch_values is not None else [n, ir.const(0, "int32")]
    call = ir.Call(
        "tirx.call_ffi_kernel",
        [ir.StringImm("compute_kernel"), b.data, a.data, step, *launch_values],
        attrs=tirx.CallFFIKernelAttr(launch_params=tags, num_kernel_args=len(parameters)),
    )
    body = [
        ir.If(
            n > 0,
            [
                ir.For(
                    step,
                    ir.const(0),
                    ir.const(3),
                    ir.const(1),
                    ir.ForKind.DEFAULT,
                    [
                        ir.If(step != 1, [ir.Evaluate(call)], []),
                    ],
                )
            ],
            [],
        )
    ]
    function = tirx.Function(
        [a, b],
        body,
        ir.TupleType([]),
        ir.DictAttrs(
            {
                "global_symbol": "main",
                "target": Target("ascend"),
            }
        ),
    )
    global_var = ir.GlobalVar("main")
    host = ir.IRModule({global_var: function}).with_attr("tl.source_host_schema", 2)
    host = host.with_attr("tl.source_host_module", ir.IRModule({global_var: function}))
    return function, host, ir.IRModule({"compute_kernel": device})


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "float8_e4m3fn", "float4_e2m1fn"])
def test_ascend_native_arguments_and_control_flow(roundtrip, dtype, tmp_path):
    function, host, device = native_launch_case(dtype=dtype)
    if roundtrip:
        function, host, device = (ir.load_json(ir.save_json(value)) for value in (function, host, device))
    program = SourceHostProgram(function, host)
    raw = program.raw_module()
    assert tirx.analysis.check_well_formed(raw)
    # Same-name parameters must keep distinct identities and the deliberate
    # call-site permutation, across low-precision host storage legalization.
    calls = []
    tvm_ffi.structural_walk(raw, (ir.Call, calls.append))
    launch = next(call for call in calls if call.op.name == "tirx.call_ffi_kernel")
    assert not launch.args[1].same_as(launch.args[2])
    source = TLAscendSourceWrapper(
        ir.IRModule({"main": function}),
        "// retained device source\n",
        Target("ascend"),
        device_mod=device,
        host_mod=host,
    ).lib_code
    assert source.startswith("// retained device source\n")
    assert source.count("compute_kernel<<<") == 1
    assert "if (" in source and "for (" in source and "!= 1" in source
    assert "aclrtGetLastError" in source and "TVM_FFI_SAFE_CALL_BEGIN" in source
    assert "cudaLaunch" not in source and "hipLaunch" not in source
    (tmp_path / "host.cpp").write_text(source)
    (tmp_path / "host.json").write_text(ir.save_json(host))


@pytest.mark.parametrize(
    "tags,values,error",
    [
        (["blockIdx.x", "blockIdx.x"], [ir.const(1), ir.const(1)], "Duplicate"),
        (["threadIdx.x"], [ir.const(32)], "no thread block"),
        (["clusterCtaIdx.x"], [ir.const(2)], "Unsupported"),
        (["blockIdx.x"], [ir.const(1.0, "float32")], "scalar integer"),
    ],
)
def test_ascend_malformed_launch_metadata_is_rejected(tags, values, error):
    function, host, device = native_launch_case(tags=tags, launch_values=values)
    with pytest.raises((ValueError, TypeError), match=error):
        SourceHostProgram(function, host).emit_device(device, Target("ascend"))
