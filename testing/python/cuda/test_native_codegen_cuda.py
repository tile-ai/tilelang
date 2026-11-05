"""Compile TileLang's CUDA emitter with the native parameter/launch contracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx._compile import _split_host_device_modules
from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline


ROOT = Path(__file__).resolve().parents[3]


def _tl_call(name, *args, dtype="void"):
    operands = [arg if isinstance(arg, tvm.ir.Expr) else tvm.ir.const(arg) for arg in args]
    return tvm.ir.Call(name, operands, ty_args=[tvm.ir.PrimType(dtype)])


def _nvcc_source(source, tmp_path, target):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA device compilation requires nvcc")
    cutlass = Path(os.environ.get("TILELANG_TEST_CUTLASS_INCLUDE", ROOT / "3rdparty/cutlass/include"))
    if not (cutlass / "cute").is_dir():
        pytest.skip("Initialize the fixed TileLang CUTLASS submodule")
    cuda_file = tmp_path / "kernel.cu"
    binary_file = tmp_path / "kernel.cubin"
    cuda_file.write_text(source)
    command = [
        nvcc,
        "-std=c++17",
        "--cubin",
        f"-arch={target.arch}",
        f"-I{ROOT / 'src'}",
        f"-I{cutlass}",
        str(cuda_file),
        "-o",
        str(binary_file),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
    (tmp_path / "nvcc.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary_file.read_bytes()


def build_cuda(func, tmp_path, *, name="main"):
    if not tvm.cuda(0).exist:
        pytest.skip("CUDA runtime/device required")
    arch = os.environ.get("TILELANG_TEST_CUDA_ARCH", "sm_" + tvm.cuda(0).compute_version.replace(".", ""))
    target = tvm.target.Target({"kind": "cuda", "arch": arch}, host="c")
    lowering, finalize_host, finalize_device = tirx_pipeline(target=target)
    lowered = lowering(tvm.IRModule({name: func.with_attr("global_symbol", name)}))
    host, devices = _split_host_device_modules(lowered)
    host = finalize_host()(host)
    (tmp_path / "lowered.py").write_text(lowered.script())
    modules, sources = [], []

    def compile_cuda(source, target, _config):
        return _nvcc_source(source, tmp_path, target)

    old = tvm_ffi.get_global_func("tilelang_callback_cuda_compile", allow_missing=True)
    tvm_ffi.register_global_func("tilelang_callback_cuda_compile", compile_cuda, override=True)
    try:
        for (device_target, _), device_mod in devices.items():
            device_mod = finalize_device()(device_mod)
            # Explicitly select TileLang's own compiler hook.  A stock TIRx
            # CUDA kernel does not test this emitter.
            module = tvm_ffi.get_global_func("target.build.tilelang_cuda")(device_mod, device_target)
            modules.append(module)
            sources.append(module.inspect_source("cuda"))
    finally:
        if old is None:
            tvm_ffi.remove_global_func("tilelang_callback_cuda_compile")
        else:
            tvm_ffi.register_global_func("tilelang_callback_cuda_compile", old, override=True)
    assert len(modules) == 1
    host_mod = tvm_ffi.get_global_func("target.build.tilelang_c")(host, target.host)
    host_mod.import_module(modules[0])
    (tmp_path / "host.cc").write_text(host_mod.inspect_source())
    library = tmp_path / "kernel.so"
    tvm.codegen.Executable(host_mod).export_library(str(library), cc="c++", options=["-std=c++17", f"-I{ROOT / 'src'}"])
    return tvm.runtime.load_module(str(library)), sources[0]


@pytest.mark.parametrize("n", [1, 33, 257])
def test_native_cuda_dynamic_add(n, tmp_path):
    size = T.dynamic("size")

    @T.function
    def add(source: T.Tensor((size,), "float32"), output: T.Tensor((size,), "float32")):
        with T.launch_thread("blockIdx.x", T.ceildiv(size, 32)) as bx, T.launch_thread("threadIdx.x", 32) as tx:
            i = bx * 32 + tx
            if i < size:
                output[i] = source[i] + T.float32(1.25)

    module, source = build_cuda(add, tmp_path)
    assert "__global__" in source and "tl_templates/cuda/reduce.h" in source
    rng = np.random.default_rng(23)
    values = rng.uniform(-4, 4, n).astype("float32")
    dev = tvm.cuda(0)
    output = tvm.runtime.tensor(np.zeros_like(values), dev)
    module(tvm.runtime.tensor(values, dev), output)
    np.testing.assert_array_equal(output.numpy(), values + np.float32(1.25))


def test_native_cuda_shared_exchange(tmp_path):
    @T.function
    def exchange(source: T.Tensor((32,), "float32"), output: T.Tensor((32,), "float32")):
        with T.launch_thread("blockIdx.x", 1) as _bx, T.launch_thread("threadIdx.x", 32) as tx:
            shared = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
            shared[tx] = source[tx]
            T.gpu_storage_sync("shared")
            output[tx] = shared[(tx + 1) % 32]

    module, source = build_cuda(exchange, tmp_path)
    assert "__shared__" in source and "__syncthreads" in source
    values = np.arange(32, dtype="float32")
    dev = tvm.cuda(0)
    output = tvm.runtime.tensor(np.zeros_like(values), dev)
    module(tvm.runtime.tensor(values, dev), output)
    np.testing.assert_array_equal(output.numpy(), np.roll(values, -1))


def test_native_cuda_cp_async_bytes(tmp_path):
    @T.function
    def copy(source: T.Tensor((32,), "float32"), output: T.Tensor((32,), "float32")):
        with T.launch_thread("blockIdx.x", 1) as _bx, T.launch_thread("threadIdx.x", 32) as tx:
            shared = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
            T.evaluate(_tl_call("tl.ptx_cp_async_bytes", T.address_of(shared[tx]), T.address_of(source[tx]), 4))
            T.evaluate(_tl_call("tl.ptx_commit_group"))
            T.evaluate(_tl_call("tl.ptx_wait_group", 0))
            T.gpu_storage_sync("shared")
            output[tx] = shared[(tx + 1) % 32]

    module, source = build_cuda(copy, tmp_path)
    assert "cp_async_gs<4>" in source
    values = np.arange(32, dtype="float32") - np.float32(19)
    dev = tvm.cuda(0)
    output = tvm.runtime.tensor(np.zeros_like(values), dev)
    module(tvm.runtime.tensor(values, dev), output)
    np.testing.assert_array_equal(output.numpy(), np.roll(values, -1))


def test_native_cuda_bulk_copy_barrier_id(tmp_path):
    if tvm.cuda(0).exist and int(tvm.cuda(0).compute_version.split(".")[0]) < 9:
        pytest.skip("cp.async.bulk requires SM90+")

    @T.function
    def bulk(source: T.Tensor((32,), "float32"), output: T.Tensor((32,), "float32")):
        with T.launch_thread("blockIdx.x", 1) as _bx, T.launch_thread("threadIdx.x", 32) as tx:
            shared = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
            T.evaluate(_tl_call("tl.create_barriers", 1))
            if tx == 0:
                T.evaluate(_tl_call("tl.ptx_init_barrier_thread_count", 0, 1))
                T.evaluate(_tl_call("tl.ptx_fence_barrier_init"))
            T.gpu_storage_sync("shared")
            if tx == 0:
                T.evaluate(_tl_call("tl.ptx_arrive_barrier_expect_tx", 0, 128))
                T.evaluate(_tl_call("tl.ptx_cp_async_bulk", T.address_of(shared[0]), 0, T.address_of(source[0]), 0, 128, 0))
                T.evaluate(_tl_call("tl.ptx_wait_barrier", 0))
            T.gpu_storage_sync("shared")
            output[tx] = shared[tx]

    module, source = build_cuda(bulk, tmp_path)
    assert "mbarrier_mem[1]" in source and "&mbarrier[0]" in source
    values = np.arange(32, dtype="float32") * np.float32(1.25)
    dev = tvm.cuda(0)
    output = tvm.runtime.tensor(np.zeros_like(values), dev)
    module(tvm.runtime.tensor(values, dev), output)
    np.testing.assert_array_equal(output.numpy(), values)


def _raw_kernel(name, statements, *, params=()):
    return tvm.tirx.Function(
        list(params), statements, ret_type=tvm.ir.PrimType("void"), attrs=tvm.ir.DictAttrs({"global_symbol": name, "calling_conv": 2})
    )


def _source_only(functions):
    return tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        tvm.IRModule(functions), tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    )


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_legacy_barrier_id_is_rejected(index):
    statements = [tvm.ir.Evaluate(_tl_call("tl.create_barriers", 2)), tvm.ir.Evaluate(_tl_call("tl.ptx_wait_barrier", index))]
    with pytest.raises(ValueError, match="barrier ID out of bounds"):
        _source_only({"main": _raw_kernel("main", statements)})


def test_legacy_barrier_requires_creation_in_current_kernel():
    first = _raw_kernel("first", [tvm.ir.Evaluate(_tl_call("tl.create_barriers", 1))])
    second = _raw_kernel("second", [tvm.ir.Evaluate(_tl_call("tl.ptx_wait_barrier", 0))])
    with pytest.raises(ValueError, match="create_barriers in the same kernel"):
        _source_only({"first": first, "second": second})


def test_source_only_does_not_invoke_compile_callback():
    called = []
    name = "tilelang_callback_cuda_compile"
    old = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda *args: called.append(args), override=True)
    try:
        module = _source_only({"first": _raw_kernel("first", [])})
        assert "void first(" in module.inspect_source("cuda")
        assert not called
    finally:
        if old is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, old, override=True)


def test_duplicate_cuda_global_symbols_are_rejected():
    with pytest.raises(ValueError, match="Duplicate CUDA kernel global_symbol"):
        _source_only({"first": _raw_kernel("same", []), "second": _raw_kernel("same", [])})
