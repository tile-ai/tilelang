"""Native ROCm and Metal tile-op producers and actual source-code consumers.

These are construction/lowering/code-generation tests. They intentionally make
no claim about device execution without an AMD GPU or an Apple Metal runner.
"""

import json
from pathlib import Path

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx._compile import _split_host_device_modules
from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

from test_native_cuda_copy import calls, function, lower
from test_native_cuda_ops import gemm
from test_native_reducer_contracts import allocate, integer, objects, region, run_pass, tensor, tileop


def backend_target(backend):
    return tvm_tirx.target.Target({"kind": "hip", "mcpu": "gfx942"} if backend == "rocm" else {"kind": "metal"})


def gpu_function(params, body, backend, threads=128):
    value = function(params, body, threads=threads).with_attr("target", backend_target(backend))
    block = ir.Var("block", "int32")
    return tirx.Function(
        value.params,
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), integer(1)], [block], {}, value.body)],
        value.ret_type,
        value.attrs,
    )


def emit_source(value, backend, output):
    """Cross the actual layout, pointer, host/device and TileLang emitter APIs."""
    value = lower(value, "native-" + backend + "-source")
    value = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": value}))["main"]
    value = run_pass("LegalizeSafeMemoryAccess", value)
    value = run_pass("LowerAccessPtr", value)
    module = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(ir.IRModule({"main": value}))
    target = value.attrs["target"].with_host(tvm_tirx.target.Target("c"))
    lowering, _, finalize_device = tirx_pipeline(target=target)
    lowered = lowering(module)
    _, devices = _split_host_device_modules(lowered)
    assert len(devices) == 1
    (((device_target, _), device_module),) = devices.items()
    device_module = finalize_device()(device_module)
    assert all(tirx.analysis.check_well_formed(function) for function in device_module.functions.values())
    suffix = "hip" if backend == "rocm" else "metal"
    build = tvm_ffi.get_global_func("target.build.tilelang_" + suffix + "_without_compile")
    result = build(device_module, device_target)
    source = result.inspect_source()
    assert source
    output = Path(output)
    (output / "lowered.json").write_text(ir.save_json(lowered))
    (output / "device.json").write_text(ir.save_json(device_module))
    (output / ("kernel." + suffix)).write_text(source)
    (output / "coverage.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "target": str(device_target),
                "ir": "passed",
                "codegen": "passed",
                "device_compile": "not_run",
                "device_execute": "not_run",
                "reason": "AMD SDK/device or Apple Metal runner is required separately",
            },
            indent=2,
        )
    )
    return source


@pytest.mark.parametrize("backend", ["rocm", "metal"])
@pytest.mark.parametrize("operation", ["fill", "copy", "transpose"])
def test_generic_gpu_ops_real_native_source(backend, operation, tmp_path):
    shape = [7, 17] if operation == "transpose" else [257]
    out_shape = [17, 7] if operation == "transpose" else shape
    source, output = tensor("A", shape, scope="global"), tensor("B", out_shape, scope="global")
    if operation == "fill":
        body = [tileop("fill", region(output), ir.FloatImm("float32", -0.0))]
    else:
        body = [tileop(operation, region(source), region(output))]
    code = emit_source(gpu_function([source, output], body, backend), backend, tmp_path)
    if backend == "metal":
        assert "[[kernel" in code and "void main_kernel" in code and "device float" in code
    else:
        assert "__global__" in code
    if operation == "fill":
        assert "-0.000000" in code or "-0.0" in code
    assert "tl.tileop" not in code


@pytest.mark.parametrize("name", ["atomicadd", "atomicmax", "atomicmin"])
@pytest.mark.parametrize("scalar", [False, True])
def test_rocm_atomic_uses_typed_pointer_and_native_memory_order(name, scalar, tmp_path):
    source, output = tensor("A", [257], scope="global"), tensor("B", [257], scope="global")
    operand = ir.FloatImm("float32", 3) if scalar else region(source)
    value = gpu_function([source, output], [tileop(name, operand, region(output), metadata={"memory_order": integer(5)})], "rocm")
    result = lower(value, f"rocm-{name}-{scalar}")
    op_name = {"atomicadd": "tl.atomic_add_elem_op", "atomicmax": "tl.atomic_max_elem_op", "atomicmin": "tl.atomic_min_elem_op"}[name]
    effects = calls(result, op_name)
    assert effects
    for call in effects:
        ir.assert_structural_equal(call.ty, ir.PrimType("void"))
        ir.assert_structural_equal(call.args[0].ty.element_type, ir.PrimType("float32"))
        assert call.args[0].ty.scope.name == "global"
        assert int(call.args[2].value) == 5
    code = emit_source(value, "rocm", tmp_path)
    assert "Atomic" in code or "atomic" in code


@pytest.mark.parametrize("managed", [False, True])
def test_rocm_async_copy_preserves_commit_wait_ownership(managed, tmp_path):
    source, output = tensor("A", [128], scope="global"), tensor("B", [128], scope="global")
    shared = tensor("S", [128], scope="shared")
    metadata = {"no_implicit_async_commit_wait": True} if managed else {"is_async_copy": True}
    body = [allocate(shared), tileop("copy", region(source), region(shared), metadata=metadata)]
    result = lower(gpu_function([source, output], body, "rocm"), "rocm-async-" + str(managed))
    assert calls(result, "tl.ptx_cp_async")
    assert bool(calls(result, "tl.ptx_commit_group")) == (not managed)
    assert not calls(result, "tl.ptx_wait_group")
    for call in calls(result, "tl.ptx_cp_async"):
        assert call.args[0].ty.scope.name == "shared"
        assert call.args[1].ty.scope.name == "global"
        assert isinstance(call.args[2].ty, ir.PrimType)


def test_rocm_explicit_async_rejects_gfx90a():
    source, shared = tensor("A", [128], scope="global"), tensor("S", [128], scope="shared")
    value = gpu_function(
        [source], [allocate(shared), tileop("copy", region(source), region(shared), metadata={"is_async_copy": True})], "rocm"
    )
    value = value.with_attr("target", tvm_tirx.target.Target({"kind": "hip", "mcpu": "gfx90a"}))
    with pytest.raises(Exception, match="Explicit async copy semantics require ROCm"):
        run_pass("LayoutInference", value)


@pytest.mark.parametrize("dtype,lanes,expected", [("float32", 1, True), ("float16", 1, False), ("float16", 2, True), ("int4", 8, True)])
def test_rocm_async_injector_element_count_and_synchronous_visibility(dtype, lanes, expected):
    source, shared = tensor("A", [lanes], dtype, "global"), tensor("S", [lanes], dtype, "shared")
    lane = ir.Var("v", "int32")
    store = ir.TensorStore(shared, [lane], ir.TensorLoad(source, [lane]))
    body = ir.SeqStmt([ir.For(lane, integer(0), integer(lanes), integer(1), ir.ForKind.VECTORIZED, [store])])
    result = tvm_ffi.get_global_func("tl.testing.InjectROCmAsyncCopy")(body, False)
    assert result["injected"] == expected
    copy_calls = calls(result["statement"], "tl.ptx_cp_async")
    assert bool(copy_calls) == expected
    if expected:
        assert len(copy_calls) == 1
        assert int(copy_calls[0].args[2].value) == 1
        assert calls(result["statement"], "tl.ptx_commit_group")
        assert calls(result["statement"], "tl.ptx_wait_group")


@pytest.mark.parametrize("mcpu,expected,warp_size", [("gfx942", "rocm.mfma", 64), ("gfx1100", "rocm.wmma", 32)])
def test_rocm_gemm_native_dispatch_and_wavefront(mcpu, expected, warp_size):
    a, b = tensor("A", [32, 32], "float16", "shared"), tensor("B", [32, 32], "float16", "shared")
    c = tensor("C", [32, 32], "float32", "local.fragment")
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(gemm(a, b, c, 32, 32, 32).value)
    target = tvm_tirx.target.Target({"kind": "hip", "mcpu": mcpu})
    assert tvm_ffi.get_global_func("tl.GemmGetGemmInstructionKey")(parsed, 128, target) == expected
    assert tvm_ffi.get_global_func("tl.TargetRocmGetWarpSize")(target) == warp_size


@pytest.mark.parametrize(
    "scope,shape,fill_name,store_name",
    [
        ("metal.simdgroup", [8, 8], "tirx.metal.make_filled_simdgroup_matrix", "tirx.metal.simdgroup_store"),
        ("metal.cooperative_tensor", [32, 32], "tl.cooperative_tensor_fill", "tl.cooperative_tensor_store"),
    ],
)
def test_metal_fragment_ops_retain_native_allocation_identity(scope, shape, fill_name, store_name, tmp_path):
    fragment = tensor("F", shape, "float32", scope)
    output = tensor("B", shape, scope="global")
    body = [
        allocate(fragment),
        tileop("fill", region(fragment), ir.FloatImm("float32", 2)),
        tileop("copy", region(fragment), region(output)),
    ]
    value = gpu_function([output], body, "metal", threads=32)
    result = lower(value, "metal-" + scope)
    fills, stores = calls(result, fill_name), calls(result, store_name)
    assert fills and stores
    bindings = {binding.var for binding in objects(result, ir.Bind)}
    for call in fills + stores:
        assert isinstance(call.args[0], ir.Var)
        assert call.args[0] in bindings
        assert isinstance(call.args[0].ty, tirx.TensorType)
        assert call.args[0].ty.scope.name == scope
    for call in stores:
        assert isinstance(call.args[2].ty, ir.PtrType)
        assert call.args[2].ty.scope.name == "global"
        assert int(call.args[3].value) == shape[-1]
    code = emit_source(value, "metal", tmp_path)
    assert "simdgroup" in code if scope == "metal.simdgroup" else "cooperative_tensor" in code
