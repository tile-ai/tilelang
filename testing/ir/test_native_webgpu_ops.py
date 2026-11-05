"""WebGPU tile ops use native tensors, SIMT lowering, and the real WGSL emitter.

Numeric checks execute the WebGPU-lowered independent per-thread operations on
the C backend after serializing launch dimensions. Browser/device execution is separate.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx._compile import _split_host_device_modules
from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

from tilelang import transform
from tilelang.testing.packed import compile_packed_function
from tilelang.webgpu import language as T
from tilelang.webgpu.codegen import build_webgpu
from tilelang.webgpu.pipeline import _lower_workgroup_storage

from test_native_ascend_ops import strided_tensor, subregion
from test_native_cuda_copy import calls, lower
from test_native_reducer_contracts import allocate, integer, objects, region, run_pass, tensor, tileop


def _target(subgroups=False):
    return tvm_tirx.target.Target({"kind": "webgpu", "supports_subgroups": subgroups})


def _function(params, body, threads=128, *, subgroups=False, thread=None):
    block = ir.Var("block", "int32")
    if thread is None:
        thread = ir.Var("thread", "int32")
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(threads)], [thread], {}, body)
    return (
        tirx.Function(
            params,
            [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), integer(1)], [block], {}, [launch])],
            ir.PrimType("void"),
        )
        .with_attr("target", _target(subgroups))
        .with_attr("global_symbol", "main")
    )


def _prepare(value, name):
    result = lower(value, name)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert tirx.analysis.check_well_formed(restored)
    assert not tirx.analysis.undefined_vars(restored)
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": result}))["main"]
    result = run_pass("LegalizeSafeMemoryAccess", result)
    result = run_pass("LowerAccessPtr", result)
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("webgpu")(ir.IRModule({"main": result}))
    assert tirx.analysis.check_well_formed(module)
    return module["main"]


def _emit(value, directory):
    directory = Path(directory)
    target = value.attrs["target"].with_host(tvm_tirx.target.Target("c"))
    lowering, _, finalize_device = tirx_pipeline(target=target)
    lowered = lowering(ir.IRModule({"main": value}))
    _, devices = _split_host_device_modules(lowered)
    assert len(devices) == 1
    (((device_target, _), device_module),) = devices.items()
    assert device_target.kind.name == "webgpu"
    device_module = finalize_device()(device_module)
    assert all(tirx.analysis.check_well_formed(function) for function in device_module.functions.values())
    device_module = _lower_workgroup_storage(device_module)
    compiled = build_webgpu(device_module, device_target)
    code = compiled.inspect_source("wgsl")
    assert "@compute" in code and "@workgroup_size(" in code
    assert "tl.tileop" not in code and "__global__" not in code
    (directory / "lowered.json").write_text(ir.save_json(lowered))
    (directory / "device.json").write_text(ir.save_json(device_module))
    (directory / "kernel.wgsl").write_text(code)
    (directory / "coverage.json").write_text(
        json.dumps(
            {
                "target": str(device_target),
                "native_lowering": "passed",
                "wgsl_source": "passed",
                "browser_compile": "not_run",
                "webgpu_execute": "not_run",
            },
            indent=2,
        )
    )
    return code


@pytest.mark.parametrize("operation", ["copy", "fill", "transpose"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "uint32"])
@pytest.mark.parametrize("subgroups", [False, True])
def test_webgpu_ops_emit_real_native_wgsl(operation, dtype, subgroups, tmp_path):
    shape = [7, 17] if operation == "transpose" else [257]
    out_shape = [17, 7] if operation == "transpose" else shape
    source, output = tensor("A", shape, dtype, "global"), tensor("B", out_shape, dtype, "global")
    if operation == "fill":
        constant = ir.FloatImm(dtype, -0.0) if dtype.startswith("float") else integer(19, dtype)
        body = [tileop("fill", region(output), constant)]
    else:
        body = [tileop(operation, region(source), region(output))]
    value = _prepare(_function([source, output], body, subgroups=subgroups), "webgpu-" + operation + "-" + dtype + "-" + str(subgroups))
    assert objects(value, ir.TensorStore)
    if operation != "fill":
        assert objects(value, ir.TensorLoad)
    code = _emit(value, tmp_path)
    assert "@workgroup_size(128" in code
    if dtype == "float16":
        assert "enable f16;" in code
    if operation == "fill" and dtype.startswith("float"):
        assert "-0.0" in code or "-0.000000" in code


@pytest.mark.parametrize("scope,size", [("shared", 257), ("local.fragment", 257), ("local", 7), ("local.var", 1)])
def test_webgpu_fill_and_copy_keep_storage_and_tail_guards(scope, size, tmp_path):
    thread = ir.Var("thread", "int32")
    private = scope in {"local", "local.var"}
    output = tensor("B", [size * 128 if private else size], "float32", "global")
    temporary = tensor("S", [size], "float32", scope)
    destination = subregion(output, [thread * size], [size]) if private else region(output)
    body = [
        allocate(temporary),
        tileop("fill", region(temporary), ir.FloatImm("float32", 3.5)),
        tileop("copy", region(temporary), destination),
    ]
    value = _prepare(_function([output], body, thread=thread), "webgpu-storage-" + scope)
    allocations = calls(value, "tirx.alloc_tensor")
    assert allocations
    if scope == "shared":
        assert any(call.ty.scope.name == "shared" for call in allocations)
    else:
        assert any(call.ty.scope.name.startswith("local") for call in allocations)
    code = _emit(value, tmp_path)
    if scope == "shared":
        assert "var<workgroup>" in code
    assert "3.5" in code or "3.500000" in code


def _serialize_launches(value):
    def replace(launch):
        assert launch.op.name == "tirx.launch_thread"
        assert launch.args[0].value in {"blockIdx.x", "threadIdx.x"}
        assert len(launch.body_params) == 1 and not launch.result_vars
        return ir.For(launch.body_params[0], integer(0), launch.args[1], integer(1), ir.ForKind.DEFAULT, launch.body)

    result = tvm_ffi.structural_map(value, (ir.RegionStmt, replace), order="post")
    assert not objects(result, ir.RegionStmt)
    result = result.with_attr("target", tvm_tirx.target.Target("c"))
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    return result


@pytest.mark.parametrize("operation", ["copy", "fill", "transpose"])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_webgpu_strided_regions_execute_correctly_when_launches_are_serialized(operation, dtype, tmp_path):
    source_storage = tensor("source_storage", [512], dtype, "global")
    output_storage = tensor("output_storage", [512], dtype, "global")
    source = strided_tensor("source_view", [6, 11], [32, 2], 7, dtype, "global")
    output_shape, output_stride, destination_start = ([9, 7], [31, 2], [2, 1]) if operation == "transpose" else ([7, 12], [40, 3], [2, 4])
    output = strided_tensor("output_view", output_shape, output_stride, 5, dtype, "global")
    extent = [5, 3] if operation == "transpose" else [3, 5]
    destination = subregion(output, destination_start, extent)
    prefix = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_storage.data], ty_args=[source.ty])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_storage.data], ty_args=[output.ty])),
    ]
    if operation == "fill":
        constant = ir.FloatImm(dtype, -0.0) if dtype == "float32" else integer(-7, dtype)
        operation_stmt = tileop("fill", destination, constant)
    else:
        operation_stmt = tileop(operation, subregion(source, [1, 2], [3, 5]), destination)
    value = _prepare(_function([source_storage, output_storage], [*prefix, operation_stmt]), "webgpu-strided-" + operation + "-" + dtype)
    _emit(value, tmp_path)
    serial = _serialize_launches(value)
    (tmp_path / "serialized.json").write_text(ir.save_json(serial))
    executable = compile_packed_function(serial)
    source_array = np.arange(512, dtype=dtype) * 3 - 17
    expected = np.full(512, -333, dtype=dtype)
    for row in range(3):
        for column in range(5):
            dst_row, dst_column = (column, row) if operation == "transpose" else (row, column)
            dst_index = 5 + (destination_start[0] + dst_row) * output_stride[0] + (destination_start[1] + dst_column) * output_stride[1]
            expected[dst_index] = (
                (-0.0 if dtype == "float32" else -7) if operation == "fill" else source_array[7 + (row + 1) * 32 + (column + 2) * 2]
            )
    runtime_source = tvm_tirx.runtime.tensor(source_array)
    runtime_output = tvm_tirx.runtime.tensor(np.full(512, -333, dtype=dtype))
    executable(runtime_source, runtime_output)
    actual = runtime_output.numpy()
    np.testing.assert_array_equal(actual, expected)
    if dtype == "float32":
        np.testing.assert_array_equal(actual.view("uint32"), expected.view("uint32"))
    np.testing.assert_array_equal(runtime_source.numpy(), source_array)


def test_webgpu_real_dialect_launch_and_ops_reach_wgsl(tmp_path):
    @T.prim_func
    def before(A: T.Tensor((7, 17), "float32"), B: T.Tensor((17, 7), "float32")):
        with T.Kernel(1, threads=64):
            T.transpose(A, B)

    value = before.with_attr("target", _target())
    materialized = transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=True, default_threads=None)(
        ir.IRModule({"main": value})
    )["main"]
    code = _emit(_prepare(materialized, "webgpu-real-dialect-transpose"), tmp_path)
    assert "@workgroup_size(64" in code


def test_webgpu_fill_preserves_unsupported_scope_diagnostic():
    unsupported = tensor("S", [16], "float32", "shared.l1")
    value = _function([], [allocate(unsupported), tileop("fill", region(unsupported), ir.FloatImm("float32", 1))])
    with pytest.raises((ValueError, RuntimeError, TypeError), match="Unsupported scope"):
        lower(value, "webgpu-unsupported-fill")
