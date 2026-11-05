"""NVRTC consumes the native TileLang lowering and executes on a real GPU."""

import shutil
import subprocess
import sys

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.engine.param import dump_kernel_params


def compile_nvrtc(function, out_idx=(1,)):
    return tilelang.compile(
        function,
        out_idx=list(out_idx),
        target={"kind": "cuda", "arch": "sm_100a"},
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="nvrtc",
        pass_configs={"tl.disable_warp_specialized": True},
    )


def static_add(value=1):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + T.float32(value)

    return main


def test_public_nvrtc_add_and_distinct_kernel_lifetimes(tmp_path):
    first = compile_nvrtc(static_add(1))
    second = compile_nvrtc(static_add(5))
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(first(data), data + 1, rtol=0, atol=0)
    torch.testing.assert_close(second(data), data + 5, rtol=0, atol=0)
    torch.testing.assert_close(first(data), data + 1, rtol=0, atol=0)
    assert first.adapter.kernels is not second.adapter.kernels
    assert "_native_host_schema = 1" in first.adapter.get_host_source()
    (tmp_path / "host.py").write_text(first.adapter.get_host_source())
    (tmp_path / "device.cu").write_text(first.get_kernel_source())


def test_public_nvrtc_shared_copy():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            T.copy(A, shared)
            T.copy(shared, B)

    kernel = compile_nvrtc(main)
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(data), data, rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, 1, 257])
def test_native_dynamic_host_control_flow_and_stream(size, tmp_path):
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

    kernel = compile_nvrtc(main)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data = torch.arange(size, dtype=torch.float32, device="cuda")
        actual = kernel(data)
        expected = data + 2
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (tmp_path / "host.py").write_text(kernel.adapter.get_host_source())


@pytest.mark.parametrize("stride,offset", [(1, 3), (2, 5)])
def test_native_nvrtc_noncontiguous_input_and_logical_offset(stride, offset):
    n, row_stride = T.dynamic("n", "int64"), T.dynamic("stride", "int64")

    @T.prim_func
    def main(A: T.Buffer((n,), "float32", strides=(row_stride,)), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < n:
                    B[index] = A[index] + T.float32(1)

    kernel = compile_nvrtc(main)
    data = torch.arange(offset + 17 * stride, dtype=torch.float32, device="cuda")[offset::stride]
    torch.testing.assert_close(kernel(data), data + 1, rtol=0, atol=0)


def test_nvrtc_explicit_stream_preserves_current_stream_and_output_order():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + T.float32(1)
                C[i] = A[i] + T.float32(2)

    kernel = compile_nvrtc(main, out_idx=(2, 1))
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    stream.wait_stream(current)
    actual = kernel(data, stream=stream.cuda_stream)
    stream.synchronize()
    assert torch.cuda.current_stream() == current
    torch.testing.assert_close(actual[0], data + 2, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], data + 1, rtol=0, atol=0)


def test_native_nvrtc_rejects_shape_and_dtype_mismatch():
    kernel = compile_nvrtc(static_add())
    with pytest.raises(ValueError, match="shape"):
        kernel(torch.zeros(127, dtype=torch.float32, device="cuda"))
    with pytest.raises(TypeError, match="expected"):
        kernel(torch.zeros(128, dtype=torch.float16, device="cuda"))


def test_nvrtc_two_kernels_with_programmatic_dependency():
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

    kernel = compile_nvrtc(main, out_idx=(1, 2))
    data = torch.arange(64, dtype=torch.float32, device="cuda")
    first, second = kernel(data)
    torch.testing.assert_close(first, data + 1, rtol=0, atol=0)
    torch.testing.assert_close(second, (data + 1) * 2, rtol=0, atol=0)


def access_policy_window(stream):
    import cuda.bindings.driver as cuda

    attribute_id = getattr(cuda.CUstreamAttrID, "CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW", None)
    if attribute_id is None:
        attribute_id = cuda.CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
    status, attribute = cuda.cuStreamGetAttribute(stream.cuda_stream, attribute_id)
    assert status == cuda.CUresult.CUDA_SUCCESS
    return attribute.accessPolicyWindow


def test_nvrtc_l2_policy_uses_requested_stream_and_resets():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            T.annotate_l2_hit_ratio({A: 0.8})
            for i in T.Parallel(128):
                B[i] = A[i] + T.float32(1)

    kernel = compile_nvrtc(main)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data = torch.arange(128, dtype=torch.float32, device="cuda")
        actual = kernel(data)
    stream.synchronize()
    torch.testing.assert_close(actual, data + 1, rtol=0, atol=0)
    assert access_policy_window(stream).num_bytes == 0


def test_native_host_exception_resets_l2_and_restores_ffi_stream():
    import tvm_ffi
    from tilelang.jit.adapter.device_py_host import HostResources, as_pointer

    stream = torch.cuda.Stream()
    data = torch.arange(128, dtype=torch.float32, device="cuda")
    device = tvm_ffi.Device(2, 0)
    original_stream = tvm_ffi.get_raw_stream(device)
    with pytest.raises(ValueError, match="host failure after L2 setup"), HostResources(stream.cuda_stream, 0) as resources:
        resources.packed("__tvm_cuda_stream_set_access_policy_window", as_pointer(data), data.numel() * data.element_size(), 0.8)
        assert access_policy_window(stream).num_bytes == data.numel() * data.element_size()
        assert tvm_ffi.get_raw_stream(device) == stream.cuda_stream
        raise ValueError("host failure after L2 setup")
    assert access_policy_window(stream).num_bytes == 0
    assert tvm_ffi.get_raw_stream(device) == original_stream


def test_nvrtc_tensor_map_uses_by_value_kernel_abi(tmp_path):
    """Echo a genuine encoded CUtensorMap to detect pointer-vs-value packing."""
    import ctypes
    import tvm_ffi
    from tilelang.engine.param import KernelParam
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
    from tilelang.jit.adapter.device_py_host import HostResources

    a = ir.Var("A", tirx.TensorType([8, 16], "float16"))
    output = ir.Var("B", tirx.TensorType([32], "uint32"))
    descriptor_type = ir.PtrType(tirx.TensorMapType())
    descriptor = ir.Var("descriptor", descriptor_type)
    config = [6, 2, 16, 8, 2, 32, 16, 8, 1, 1, 0, 0, 0, 0]
    initialization = tirx.call_packed(
        "__tvm_tensormap_create_tiled", descriptor, *config[:2], tirx.tensor_data_ptr(a), *config[2:], ty="int32"
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
    # NVRTC injects TileLang's nvrtc_std.h, which owns its CUtensorMap type.
    # cuda.h is a host header and would require unavailable libc headers here.
    code = """extern "C" __global__ void echo_map(const __grid_constant__ CUtensorMap descriptor, unsigned* out) {
  out[threadIdx.x] = reinterpret_cast<const unsigned*>(&descriptor)[threadIdx.x];
}
"""
    adapter = NVRTCKernelAdapter(
        [KernelParam.from_var(a), KernelParam.from_var(output)],
        [1],
        {"kind": "cuda", "arch": "sm_100a"},
        function,
        host_mod=host,
        device_mod=ir.IRModule({"echo_map": device_function}),
        device_kernel_source=code,
        verbose=True,
    )
    data = torch.arange(128, dtype=torch.float16, device="cuda").reshape(8, 16)
    actual = adapter(data)
    with HostResources(0, 0) as resources:
        descriptor_memory = resources.allocate("tensormap", 1)
        tvm_ffi.get_global_func("__tvm_tensormap_create_tiled")(
            descriptor_memory, *config[:2], ctypes.c_void_p(data.data_ptr()), *config[2:]
        )
        expected = torch.tensor(list((ctypes.c_uint32 * 32).from_address(descriptor_memory.value)), dtype=torch.uint32)
    assert torch.equal(actual.cpu(), expected)
    (tmp_path / "tensormap-host.py").write_text(adapter.get_host_source())


def test_nvrtc_cache_loads_in_independent_process_without_recompiling(tmp_path):
    function = static_add(3)
    kernel = compile_nvrtc(function)
    adapter = kernel.adapter
    (tmp_path / "function.json").write_text(ir.save_json(function))
    (tmp_path / "params.json").write_text(dump_kernel_params(adapter.params))
    shutil.copy2(adapter.libpath, tmp_path / "kernel.cubin")
    shutil.copy2(str(adapter.libpath).replace(".cubin", ".py"), tmp_path / "kernel.py")
    child = tmp_path / "reload.py"
    child.write_text("""from pathlib import Path
import sys
import torch
from tvm_tirx import ir
from tilelang.engine.param import load_kernel_params
from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
from tilelang.jit.adapter.base import CachedTextSource
p = Path(__file__).parent
adapter = NVRTCKernelAdapter.from_database(
    load_kernel_params((p / "params.json").read_text()), [1], {"kind": "cuda", "arch": "sm_100a"},
    ir.load_json((p / "function.json").read_text()), CachedTextSource(path=str(p / "kernel.py")),
    CachedTextSource(), str(p / "kernel.cubin"),
)
x = torch.arange(128, dtype=torch.float32, device="cuda")
torch.testing.assert_close(adapter(x), x + 3, rtol=0, atol=0)
assert "tvm" not in sys.modules
print("fresh-process native NVRTC numerical oracle passed")
""")
    result = subprocess.run([sys.executable, str(child)], capture_output=True, text=True)
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
