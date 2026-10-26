import ctypes
import os

import pytest
from tvm_tirx import ir, tirx

if os.environ.get("TILELANG_LIGHT_IMPORT") == "1":
    pytest.skip("Raw host adapters require the full TileLang frontend", allow_module_level=True)

from tilelang.jit.adapter.native import _with_cuda_stream
from tilelang.jit.adapter.nvrtc import runtime as host_runtime

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_native_host_packed_calls_marshal_typed_pointers(monkeypatch):
    pointer = ir.Var("pointer", ir.PtrType("void"))
    stream = ir.Var("stream", ir.PtrType("void"))
    call = ir.Call("tirx.call_packed", ["test.host_pointer", pointer, stream], ty="int32")
    function = tirx.Function([pointer, stream], ir.Return(call), ret_type=ir.PrimType("int32"))

    def callback(base, cuda_stream):
        assert isinstance(base, ctypes.c_void_p)
        assert isinstance(cuda_stream, ctypes.c_void_p)
        assert base.value == 4096
        assert cuda_stream.value == 8192
        return 37

    native_get = host_runtime.tvm_ffi.get_global_func
    monkeypatch.setattr(host_runtime.tvm_ffi, "get_global_func", lambda name: callback if name == "test.host_pointer" else native_get(name))
    program = host_runtime.NativeHostProgram(ir.save_json(function))
    assert program({}, [4096], stream=8192) == 37


def test_native_host_packed_pointer_results_are_normalized(monkeypatch):
    call = ir.Call("tirx.call_packed", ["test.host_pointer"], ty=ir.PtrType("void"))
    execution = host_runtime._Execution({}, {})
    native_get = host_runtime.tvm_ffi.get_global_func
    monkeypatch.setattr(
        host_runtime.tvm_ffi,
        "get_global_func",
        lambda name: (lambda: ctypes.c_void_p(4096)) if name == "test.host_pointer" else native_get(name),
    )
    assert execution.expression(call) == 4096


def test_host_stream_queries_use_the_explicit_adapter_stream():
    stream = ir.Var("stream", ir.PtrType("void"))
    query = ir.Call("tirx.call_packed", ["runtime.get_cuda_stream"], ty=ir.PtrType("void"))
    assert _with_cuda_stream(query, stream).same_as(stream)


def test_packed_tensor_map_initialization_has_aligned_stable_storage(monkeypatch):
    pytest.importorskip("cuda.bindings.driver")
    descriptor = ir.Var("descriptor", ir.PtrType(tirx.TensorMapType()))
    storage = host_runtime._TensorMap()
    pattern = bytes(range(128))

    def encode(pointer):
        assert isinstance(pointer, ctypes.c_void_p)
        assert pointer.value % 128 == 0
        ctypes.memmove(pointer, pattern, len(pattern))
        return 0

    native_get = host_runtime.tvm_ffi.get_global_func
    monkeypatch.setattr(
        host_runtime.tvm_ffi,
        "get_global_func",
        lambda name: encode if name == "__tvm_tensormap_create_im2col" else native_get(name),
    )
    execution = host_runtime._Execution({}, {descriptor: storage})
    call = ir.Call("tirx.call_packed", ["__tvm_tensormap_create_im2col", descriptor], ty="int32")
    assert execution.expression(call) == 0
    assert ctypes.string_at(storage.value.getPtr(), 128) == pattern
    assert storage.packed_pointer().value == storage.address


def _cuda_frontend():
    if os.environ.get("TILELANG_LIGHT_IMPORT") == "1":
        pytest.skip("Host-service execution requires the full frontend")
    import torch
    import tilelang
    import tilelang.language as language

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    return torch, tilelang, language


@pytest.mark.gpu
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_l2_host_service_executes_on_nondefault_stream(execution_backend):
    torch, tilelang, T = _cuda_frontend()

    @T.prim_func
    def increment(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=32):
            T.annotate_l2_hit_ratio({source: 0.5})
            for index in T.Parallel(64):
                output[index] = source[index] + 1.0

    kernel = tilelang.compile(increment, out_idx=[1], execution_backend=execution_backend)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randn(64, device="cuda")
        torch.testing.assert_close(kernel(source), source + 1.0)
    stream.synchronize()


@pytest.mark.gpu
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_im2col_tensor_map_host_service_executes(execution_backend):
    torch, tilelang, T = _cuda_frontend()
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA im2col requires Hopper or newer")

    @T.prim_func
    def unfold(source: T.Tensor((1, 8, 8, 32), "float16"), output: T.Tensor((64, 288), "float16")):
        with T.Kernel(4, 9, threads=128) as (pixel_block, filter_block):
            shared = T.alloc_shared((16, 32), "float16")
            T.im2col(source, shared, pixel_block, filter_block, 3, 1, 1, 1)
            T.copy(shared, output[pixel_block * 16, filter_block * 32])

    kernel = tilelang.compile(unfold, out_idx=[1], execution_backend=execution_backend)
    source = torch.randn((1, 8, 8, 32), dtype=torch.float16, device="cuda")
    expected = torch.nn.functional.unfold(source.permute(0, 3, 1, 2), 3, padding=1)
    expected = expected.reshape(1, 32, 3, 3, 64).permute(0, 4, 2, 3, 1).reshape(64, 288)
    torch.testing.assert_close(kernel(source), expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_cython_compiles_postprocessed_source_once():
    import tvm_ffi

    torch, tilelang, T = _cuda_frontend()

    @T.prim_func
    def increment(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=32):
            for index in T.Parallel(64):
                output[index] = source[index] + 1.0

    callback_name = "tilelang_callback_cuda_postproc"
    previous = tvm_ffi.get_global_func(callback_name, allow_missing=True)
    invocations = []

    def postprocess(source, target):
        assert "0x1p+0f" in source
        invocations.append(target)
        return source.replace("0x1p+0f", "0x1p+1f")

    tvm_ffi.register_global_func(callback_name, postprocess, override=True)
    try:
        kernel = tilelang.compile(increment, out_idx=[1], execution_backend="cython")
        source = torch.randn(64, device="cuda")
        torch.testing.assert_close(kernel(source), source + 2.0)
        assert len(invocations) == 1
    finally:
        if previous is None:
            tvm_ffi.registry.remove_global_func(callback_name)
        else:
            tvm_ffi.register_global_func(callback_name, previous, override=True)
