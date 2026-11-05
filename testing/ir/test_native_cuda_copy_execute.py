"""Real CUDA Copy lowering, TileLang emitter, NVCC and numerical execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx._compile import _split_host_device_modules
from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

from test_native_cuda_copy import function, lower
from test_native_reducer_contracts import allocate, integer, region, run_pass, tensor, tileop


ROOT = Path(__file__).resolve().parents[2]


def build_cuda_copy(value, output):
    """Use native host/device boundaries and the actual TileLang CUDA emitter."""
    output.mkdir(parents=True, exist_ok=True)
    target = value.attrs["target"].with_host(tvm_tirx.target.Target("c"))
    value = run_pass("LowerAccessPtr", value)
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("cuda")(ir.IRModule({"main": value}))
    lowering, finalize_host, finalize_device = tirx_pipeline(target=target)
    lowered = lowering(module)
    host, devices = _split_host_device_modules(lowered)
    host = finalize_host()(host)
    (output / "lowered.json").write_text(ir.save_json(lowered))
    (output / "lowered.py").write_text(lowered.script())
    include = Path(os.environ.get("TILELANG_TEST_INCLUDE_ROOT", ROOT / "src")).resolve()
    cutlass = Path(os.environ.get("TILELANG_TEST_CUTLASS_INCLUDE", ROOT / "3rdparty/cutlass/include")).resolve()
    assert (include / "tl_templates/cuda/reduce.h").is_file()
    assert (cutlass / "cute").is_dir()
    nvcc = shutil.which("nvcc")
    assert nvcc is not None
    compilation = []

    def compile_source(source, device_target, _config):
        source_file = output / "kernel.cu"
        binary = output / "kernel.cubin"
        source_file.write_text(source)
        command = [
            nvcc,
            "-std=c++17",
            "--cubin",
            "-arch=" + device_target.arch,
            "-I" + str(include),
            "-I" + str(cutlass),
            str(source_file),
            "-o",
            str(binary),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (output / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
        (output / "nvcc.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        compilation.append(source)
        return binary.read_bytes()

    callback = "tilelang_callback_cuda_compile"
    previous = tvm_ffi.get_global_func(callback, allow_missing=True)
    tvm_ffi.register_global_func(callback, compile_source, override=True)
    device_modules = []
    try:
        for (device_target, _), device_module in devices.items():
            device_module = finalize_device()(device_module)
            device_modules.append(tvm_ffi.get_global_func("target.build.tilelang_cuda")(device_module, device_target))
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(callback)
        else:
            tvm_ffi.register_global_func(callback, previous, override=True)
    assert len(device_modules) == len(compilation) == 1
    host_module = tvm_ffi.get_global_func("target.build.tilelang_c")(host, target.host)
    host_module.import_module(device_modules[0])
    (output / "host.cc").write_text(host_module.inspect_source())
    library = output / "kernel.so"
    tvm_tirx.codegen.Executable(host_module).export_library(str(library), cc="c++", options=["-std=c++17", "-I" + str(include)])
    return tvm_tirx.runtime.load_module(str(library)), compilation[0]


@pytest.mark.parametrize(
    "dtype,size,instruction",
    [
        ("float32", 1, "sync"),
        ("float32", 33, "sync"),
        ("float32", 257, "sync"),
        ("float16", 257, "sync"),
        ("uint8", 129, "sync"),
        ("float32", 128, "cp_async"),
        ("float32", 257, "cp_async"),
        ("float16", 256, "cp_async"),
        ("float16", 512, "cp_async"),
        ("float16", 1024, "cp_async"),
        ("uint8", 512, "cp_async"),
    ],
)
def test_cuda_copy_end_to_end(dtype, size, instruction, tmp_path):
    if not tvm_tirx.cuda(0).exist:
        pytest.skip("CUDA runtime and GPU required for numerical execution")
    source, destination = tensor("A", [size], dtype, "global"), tensor("B", [size], dtype, "global")
    shared = tensor("shared", [size], dtype, "shared")
    body = [
        allocate(shared),
        tileop("copy", region(source), region(shared), metadata={"prefer_instruction": instruction}),
        ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")])),
        tileop("copy", region(shared), region(destination), metadata={"prefer_instruction": "sync"}),
    ]
    arch = os.environ.get("TILELANG_TEST_CUDA_ARCH", "sm_" + tvm_tirx.cuda(0).compute_version.replace(".", ""))
    value = function([source, destination], body, arch=arch)
    block = ir.Var("blockIdx_x", "int32")
    value = tirx.Function(
        value.params,
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), integer(1)], [block], {}, value.body)],
        ir.NoneType(),
        attrs=value.attrs,
    )
    value = lower(value, f"cuda-exec-{dtype}-{size}-{instruction}")
    value = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": value}))["main"]
    value = run_pass("LegalizeSafeMemoryAccess", value)
    output = tmp_path.resolve()
    module, device_source = build_cuda_copy(value, output)
    assert "__global__" in device_source and "__shared__" in device_source
    if instruction == "cp_async":
        assert "cp_async_gs" in device_source
    rng = np.random.default_rng(7421)
    values = rng.integers(0, 127, size).astype(dtype) if dtype == "uint8" else rng.uniform(-3, 3, size).astype(dtype)
    dev = tvm_tirx.cuda(0)
    result = tvm_tirx.runtime.tensor(np.zeros_like(values), dev)
    module(tvm_tirx.runtime.tensor(values, dev), result)
    actual = result.numpy()
    np.testing.assert_array_equal(actual, values)
    (output / "numeric.json").write_text(
        json.dumps(
            {
                "dtype": dtype,
                "shape": [size],
                "seed": 7421,
                "oracle": "bitwise numpy array equality",
                "gpu": str(dev),
                "compute_version": dev.compute_version,
                "instruction": instruction,
                "passed": True,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("lanes,order", [(2, 0), (4, 5)])
def test_vector_atomic_uses_native_pointer_pair_on_device(lanes, order, tmp_path):
    if not tvm_tirx.cuda(0).exist:
        pytest.skip("CUDA runtime and GPU required for numerical execution")
    from test_native_vectorize_intrinsics import pointer

    size = 128 * lanes
    source = tensor("A", [size], "float32", "global")
    destination = tensor("B", [size], "float32", "global")
    thread, block, lane = [ir.Var(name, "int32") for name in ("thread", "block", "lane")]
    index = thread * lanes + lane
    call = ir.Call("tl.atomic_add_elem_op", [pointer(destination, index, mask=3), ir.TensorLoad(source, [index]), integer(order)])
    loop = ir.For(lane, integer(0), integer(lanes), integer(1), ir.ForKind.VECTORIZED, [ir.Evaluate(call)])
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("blockIdx.x"), integer(1)],
        [block],
        {},
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(128)], [thread], {}, [loop])],
    )
    arch = os.environ.get("TILELANG_TEST_CUDA_ARCH", "sm_" + tvm_tirx.cuda(0).compute_version.replace(".", ""))
    value = (
        tirx.Function([source, destination], [body], ir.NoneType())
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
        # The two independently allocated runtime inputs do not overlap.
        .with_attr("tirx.noalias", True)
    )
    value = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": value}))["main"]
    module, device_source = build_cuda_copy(value, tmp_path.resolve())
    assert f"AtomicAddx{lanes}(" in device_source
    rng = np.random.default_rng(7421)
    values = rng.integers(-128, 128, size).astype("float32")
    dev = tvm_tirx.cuda(0)
    result = tvm_tirx.runtime.tensor(np.ones_like(values), dev)
    module(tvm_tirx.runtime.tensor(values, dev), result)
    np.testing.assert_array_equal(result.numpy(), values + 1)
    (tmp_path / "numeric.json").write_text(
        json.dumps(
            {
                "lanes": lanes,
                "memory_order": order,
                "shape": [size],
                "seed": 7421,
                "oracle": "exact float32 integer contributions plus one",
                "compute_version": dev.compute_version,
                "passed": True,
            },
            indent=2,
        )
    )
