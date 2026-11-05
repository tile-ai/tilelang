"""Real native CuTeDSL host, raw device ABI, launch modes and cached cubins."""

import ast
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.engine.param import KernelParam, dump_kernel_params
from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
from tilelang.jit.adapter.device_py_host import HostResources


TARGET = {"kind": "cutedsl", "arch": "sm_100a"}


def compile_cutedsl(function, out_idx=(1,)):
    return tilelang.compile(
        function,
        out_idx=list(out_idx),
        target=TARGET,
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cutedsl",
        pass_configs={"tl.disable_warp_specialized": True},
    )


def static_add(value=1):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + T.float32(value)

    return main


@pytest.mark.parametrize("operation", ["load", "store", "reduce"])
@pytest.mark.parametrize("typed", [True, False])
def test_cutedsl_single_coordinate_is_tuple_only_for_tensor_map(operation, typed):
    """A rank-one TensorMap coordinate and bulk byte count have distinct ABIs."""
    descriptor = ir.Var("descriptor", ir.PtrType(tirx.TensorMapType()) if typed else ir.PtrType("float32"))
    shared = ir.Var("shared", ir.PtrType("float32", "shared"))
    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    value = ir.const(0 if typed else 128)
    if operation == "load":
        args = [descriptor, barrier, shared, value, ir.const(0)]
    else:
        args = [descriptor, shared, value, ir.const(int(operation == "reduce")), ir.const(0)]
    call = ir.Call("tl.tma_load" if operation == "load" else "tl.tma_store", args, ty="void")
    function = tirx.Function(
        [descriptor, shared, barrier],
        [ir.Evaluate(call)],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "native_tma", "calling_conv": 2}),
    )
    from tvm_tirx.target import Target

    runtime = tvm_ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(
        ir.IRModule({"native_tma": function}),
        Target({"kind": "cuda", "arch": "sm_100a"}),
    )
    tree = ast.parse(runtime.inspect_source("cutedsl"))
    emitted = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tma_" + operation
    )
    coordinate = emitted.args[3 if operation == "load" else 2]
    if typed:
        assert isinstance(coordinate, ast.Tuple) and len(coordinate.elts) == 1
        assert coordinate.elts[0].value == 0
    else:
        assert isinstance(coordinate, ast.Constant) and coordinate.value == 128


def native_swizzle_source(dimensions, order):
    from tvm_tirx.target import Target

    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
        {
            "node": None,
            "key": "threadblock_swizzle_pattern",
            "value": ir.Tuple([ir.StringImm("rasterization2D" + order), ir.const(8)]),
        }
    )
    region = ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [])
    function = tirx.Function(
        [],
        [region],
        ir.PrimType("void"),
        ir.DictAttrs(
            {
                "global_symbol": "native_swizzle",
                "calling_conv": 2,
                "cuda.launch_dimensions": [ir.const(value) for value in dimensions],
            }
        ),
    )
    runtime = tvm_ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(
        ir.IRModule({"native_swizzle": function}),
        Target({"kind": "cuda", "arch": "sm_100a"}),
    )
    return runtime.inspect_source("cutedsl")


@pytest.mark.parametrize("order", ["Row", "Column"])
@pytest.mark.parametrize("cluster", [1, 2])
def test_cutedsl_native_cluster_dimensions_select_rasterization(order, cluster):
    source = native_swizzle_source([32, 1, 1, cluster, 1, 1], order)
    tree = ast.parse(source)
    emitted = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("rasterization2D")
    )
    assert emitted.func.attr == "rasterization2D" + order + ("WithCluster" if cluster > 1 else "")
    assert [argument.value for argument in emitted.args] == ([4, 2] if cluster > 1 else [8])


@pytest.mark.parametrize("dimensions", [[32, 1, 1, 2, 1], [32, 1, 1, -1, 1, 1], [32, 1, 1, 2, 2, 1]])
def test_cutedsl_native_rasterization_rejects_invalid_cluster_dimensions(dimensions):
    with pytest.raises(ValueError):
        native_swizzle_source(dimensions, "Row")


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_public_cutedsl_dynamic_host_and_explicit_stream(mode, monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_TVMFFI", "1")
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        if n > 0:
            for step in T.serial(3):
                if step != 1:
                    with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
                        for i in T.Parallel(128):
                            index = block * 128 + i
                            if index < n:
                                B[index] = A[index] + T.cast(step, "float32")

    kernel = compile_cutedsl(main)
    stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
    # Same compiled adapter: the first live shape and loop scalar must not be
    # embedded into a CuTe specialization. Zero follows the real host branch.
    for size in (0, 1, 257):
        with torch.cuda.stream(stream):
            data = torch.arange(size, dtype=torch.float32, device="cuda")
        actual = kernel(data, stream=stream)
        stream.synchronize()
        torch.testing.assert_close(actual, data + 2, rtol=0, atol=0)
        assert torch.cuda.current_stream() == current
    assert len(kernel.adapter.lib_generator.runtime._entries) == 1
    (tmp_path / "host.py").write_text(kernel.get_host_source())
    (tmp_path / "device.py").write_text(kernel.get_kernel_source())


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_distinct_modules_and_cleanup(mode, monkeypatch):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    first, second = compile_cutedsl(static_add(1)), compile_cutedsl(static_add(5))
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(first(data), data + 1, rtol=0, atol=0)
    torch.testing.assert_close(second(data), data + 5, rtol=0, atol=0)
    first.adapter.cleanup()
    torch.testing.assert_close(second(data), data + 5, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="cleaned up"):
        first(data)
    reopened = compile_cutedsl(static_add(1))
    assert reopened is not first
    torch.testing.assert_close(reopened(data), data + 1, rtol=0, atol=0)
    torch.testing.assert_close(second(data), data + 5, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="cleaned up"):
        first(data)


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_shared_copy(mode, monkeypatch):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            T.copy(A, shared)
            T.copy(shared, B)

    kernel = compile_cutedsl(main)
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(data), data, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_cluster_launch_matches_hardware_rank(mode, monkeypatch):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    @T.prim_func
    def main(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.ClusterKernel(8, cluster_dims=2, threads=32) as block:
            T.cluster_arrive()
            T.cluster_wait()
            if T.get_thread_binding() == 0:
                B[block] = A[block] + T.block_rank_in_cluster()
            T.cluster_sync()

    kernel = compile_cutedsl(main)
    data = torch.arange(8, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(kernel(data), data + data % 2, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_l2_policy_uses_requested_stream_and_resets(mode, monkeypatch):
    import cuda.bindings.driver as cuda

    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            T.annotate_l2_hit_ratio({A: 0.8})
            for i in T.Parallel(128):
                B[i] = A[i] + T.float32(1)

    kernel = compile_cutedsl(main)
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        data = torch.arange(128, dtype=torch.float32, device="cuda")
    actual = kernel(data, stream=stream)
    stream.synchronize()
    torch.testing.assert_close(actual, data + 1, rtol=0, atol=0)
    assert torch.cuda.current_stream() == current
    attribute_id = getattr(cuda.CUstreamAttrID, "CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW", None)
    if attribute_id is None:
        attribute_id = cuda.CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
    status, attribute = cuda.cuStreamGetAttribute(stream.cuda_stream, attribute_id)
    assert status == cuda.CUresult.CUDA_SUCCESS
    assert attribute.accessPolicyWindow.num_bytes == 0


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_strided_input_with_logical_offset(mode, monkeypatch):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    n, stride = T.dynamic("n", "int64"), T.dynamic("stride", "int64")

    @T.prim_func
    def main(A: T.Buffer((n,), "float32", strides=(stride,)), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < n:
                    B[index] = A[index] + T.float32(1)

    kernel = compile_cutedsl(main)
    for step, offset in ((1, 3), (2, 5)):
        data = torch.arange(offset + 17 * step, dtype=torch.float32, device="cuda")[offset::step]
        torch.testing.assert_close(kernel(data), data + 1, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_pdl_two_kernels_preserve_output_order(mode, monkeypatch):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64):
                B[i] = A[i] + T.float32(1)
            T.pdl_trigger()
        with T.Kernel(1, threads=64):
            T.pdl_sync()
            for i in T.Parallel(64):
                C[i] = B[i] * T.float32(2)

    kernel = compile_cutedsl(main, out_idx=(2, 1))
    data = torch.arange(64, dtype=torch.float32, device="cuda")
    second, first = kernel(data)
    torch.testing.assert_close(first, data + 1, rtol=0, atol=0)
    torch.testing.assert_close(second, (data + 1) * 2, rtol=0, atol=0)
    assert len(kernel.adapter.lib_generator.runtime._entries) == 2


def test_cutedsl_rejects_shape_dtype_and_stale_host(tmp_path):
    kernel = compile_cutedsl(static_add())
    with pytest.raises(ValueError, match="shape"):
        kernel(torch.zeros(127, dtype=torch.float32, device="cuda"))
    with pytest.raises(TypeError, match="expected"):
        kernel(torch.zeros(128, dtype=torch.float16, device="cuda"))
    stale = tmp_path / "kernel.py"
    stale.write_text("_native_host_schema = 0\n")
    with pytest.raises(ValueError, match="Stale CuTeDSL host cache"):
        CuTeDSLKernelAdapter.from_database(
            kernel.adapter.params,
            [1],
            TARGET,
            kernel.adapter.prim_func,
            "",
            "",
            str(stale),
        )


@pytest.mark.parametrize("mode", ["cpp", "cutlass"])
def test_cutedsl_native_tensor_map_by_value(mode, monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", mode)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_TVMFFI", "1")
    a = ir.Var("A", tirx.TensorType([8, 16], "float16"))
    output = ir.Var("B", tirx.TensorType([32], "uint32"))
    descriptor_type = ir.PtrType(tirx.TensorMapType())
    descriptor = ir.Var("descriptor", descriptor_type)
    config = [6, 2, 16, 8, 2, 32, 16, 8, 1, 1, 0, 0, 0, 0]
    initialization = tirx.call_packed(
        "__tvm_tensormap_create_tiled",
        descriptor,
        *config[:2],
        tirx.tensor_data_ptr(a),
        *config[2:],
        ty="int32",
    )
    launch = ir.Call(
        "tirx.call_ffi_kernel",
        [ir.StringImm("echo_map"), descriptor, tirx.tensor_data_ptr(output), ir.const(1), ir.const(32)],
        attrs=tirx.CallFFIKernelAttr(launch_fields=["grid.x", "block.x"], num_kernel_args=2),
        ty="int32",
    )
    function = tirx.Function(
        [a, output],
        [
            ir.Bind(descriptor, ir.Call("tirx.stack_alloca", [ir.StringImm("tensormap"), ir.const(1)], ty=descriptor_type)),
            ir.Evaluate(initialization),
            ir.Evaluate(launch),
        ],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "main"}),
    )
    global_var = ir.GlobalVar("main")
    host = (
        ir.IRModule({global_var: function})
        .with_attr("tl.source_host_schema", 2)
        .with_attr("tl.source_host_module", ir.IRModule({global_var: function}))
    )
    device_function = tirx.Function(
        [ir.Var("map", descriptor_type), ir.Var("out", ir.PtrType("uint32"))],
        [],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "echo_map"}),
    )
    # The wrapper supplies GridConstant annotations from the device Function;
    # the device source intentionally has no hand-added descriptor annotation.
    code = """@cute.kernel
def echo_map(descriptor, output):
    idx, _, _ = cute.arch.thread_idx()
    source = cutlass.Pointer(descriptor.get_ptr(), dtype=cutlass.Uint32)
    out = tl.make_tensor(output.iterator, (32,))
    out[idx] = source[idx]
"""
    adapter = CuTeDSLKernelAdapter(
        [KernelParam.from_var(a), KernelParam.from_var(output)],
        [1],
        TARGET,
        function,
        host_mod=host,
        device_mod=ir.IRModule({"echo_map": device_function}),
        device_kernel_source=code,
    )
    data = torch.arange(128, dtype=torch.float16, device="cuda").reshape(8, 16)
    actual = adapter(data)
    with HostResources(0, 0) as resources:
        memory = resources.allocate("tensormap", 1)
        tvm_ffi.get_global_func("__tvm_tensormap_create_tiled")(
            memory,
            *config[:2],
            ctypes.c_void_p(data.data_ptr()),
            *config[2:],
        )
        expected = torch.tensor(list((ctypes.c_uint32 * 32).from_address(memory.value)), dtype=torch.uint32)
    assert torch.equal(actual.cpu(), expected)
    (tmp_path / "host.py").write_text(adapter.get_host_source())


def test_cutedsl_fresh_process_cache_uses_provider_without_cutlass(tmp_path):
    function = static_add(3)
    kernel = compile_cutedsl(function)
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(data), data + 3, rtol=0, atol=0)
    adapter = kernel.adapter
    (tmp_path / "function.json").write_text(ir.save_json(function))
    (tmp_path / "params.json").write_text(dump_kernel_params(adapter.params))
    shutil.copy2(adapter.libpath, tmp_path / "kernel.py")
    adapter.lib_generator.runtime.export_artifacts(tmp_path)
    child = tmp_path / "reload.py"
    child.write_text("""from pathlib import Path
import os
import sys
import torch
from tvm_tirx import ir
from tilelang.engine.param import load_kernel_params
from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
from tilelang.jit.adapter.base import CachedTextSource
os.environ["TILELANG_CUTEDSL_HOST_LAUNCHER"] = "cpp"
p = Path(__file__).parent
assert "cutlass" not in sys.modules
adapter = CuTeDSLKernelAdapter.from_database(
    load_kernel_params((p / "params.json").read_text()), [1], {"kind": "cutedsl", "arch": "sm_100a"},
    ir.load_json((p / "function.json").read_text()), CachedTextSource(path=str(p / "kernel.py")),
    CachedTextSource(), str(p / "kernel.py"),
)
x = torch.arange(128, dtype=torch.float32, device="cuda")
torch.testing.assert_close(adapter(x), x + 3, rtol=0, atol=0)
assert "tvm" not in sys.modules
assert "cutlass" not in sys.modules
print("fresh-process CuTeDSL cached provider numerical oracle passed without CUTLASS import")
""")
    result = subprocess.run([sys.executable, str(child)], capture_output=True, text=True)
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    (manifest_path,) = (tmp_path / "kernel.cutedsl").glob("*/manifest.json")
    metadata = json.loads(manifest_path.read_text())
    assert metadata["schema"] == 1
    assert metadata["symbol"]
    binary = manifest_path.with_name("kernel.cubin")
    binary.write_bytes(binary.read_bytes() + b"corrupted-cache")
    restored = CuTeDSLKernelAdapter.from_database(adapter.params, [1], TARGET, function, "", "", str(tmp_path / "kernel.py"))
    with pytest.raises(ValueError, match="Invalid CuTeDSL artifact"):
        restored(data)


def test_cutedsl_public_cache_persists_later_host_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", "cpp")
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        if n < 64:
            with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
                for i in T.Parallel(128):
                    index = block * 128 + i
                    if index < n:
                        B[index] = A[index] + T.float32(1)
        else:
            with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
                for i in T.Parallel(128):
                    index = block * 128 + i
                    if index < n:
                        B[index] = A[index] + T.float32(7)

    kernel = compile_cutedsl(main)
    cache_path = Path(kernel.adapter._cache_path)
    assert not list(cache_path.glob("kernel.cutedsl/*/manifest.json"))
    for count, size, increment in ((1, 32, 1), (2, 129, 7)):
        data = torch.arange(size, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(kernel(data), data + increment, rtol=0, atol=0)
        assert len(list(cache_path.glob("kernel.cutedsl/*/manifest.json"))) == count
    (tmp_path / "function.json").write_text(ir.save_json(main))
    (tmp_path / "cache-path.txt").write_text(str(cache_path))
    child = tmp_path / "reload_public.py"
    child.write_text("""from pathlib import Path
import sys
import torch
import tilelang
from tvm_tirx import ir
p = Path(__file__).parent
assert "cutlass" not in sys.modules
kernel = tilelang.compile(
    ir.load_json((p / "function.json").read_text()), out_idx=[1],
    target={"kind": "cutedsl", "arch": "sm_100a"},
    target_host={"kind": "llvm", "mcpu": "generic"}, execution_backend="cutedsl",
    pass_configs={"tl.disable_warp_specialized": True},
)
assert Path(kernel.adapter.libpath).resolve() == (Path((p / "cache-path.txt").read_text()) / "kernel.py").resolve()
for size, increment in ((32, 1), (129, 7), (17, 1), (257, 7)):
    data = torch.arange(size, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(data), data + increment, rtol=0, atol=0)
assert "cutlass" not in sys.modules
assert "tvm" not in sys.modules
print("public cache reload: both delayed branches and new shapes execute without CUTLASS import")
""")
    result = subprocess.run([sys.executable, str(child)], capture_output=True, text=True)
    (tmp_path / "reload-public.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
