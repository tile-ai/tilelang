import sys

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.engine.lower import device_codegen, extrac_params, get_device_call, get_host_call, host_codegen
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter
from tilelang.jit.adapter.utils import parse_function_call_args


def test_parse_function_call_args_uses_host_expression_for_scalars():
    host_expression = object()
    call_args = parse_function_call_args(
        "extern \"C\" __global__ void kernel(int n)",
        [{"name": "n", "type": "ctypes.c_int32"}],
        [host_expression],
        transform_arg=lambda name, arg_type: (name, arg_type),
        fallback_arg=lambda expression: (expression, "ctypes.c_int32"),
    )
    assert call_args == [(host_expression, "ctypes.c_int32")]


@pytest.fixture
def nvrtc_ffi(monkeypatch, tmp_path):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    from cuda.pathfinder import DynamicLibNotFoundError

    try:
        nvrtc.nvrtcVersion()
    except DynamicLibNotFoundError:
        pytest.skip("NVRTC library is unavailable")
    from tilelang.contrib import nvcc
    from tilelang.env import env

    def unexpected_nvcc(*args, **kwargs):
        pytest.fail("NVCC was invoked during NVRTC device compilation")

    monkeypatch.setattr(nvcc, "compile_cuda", unexpected_nvcc)
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    return {tilelang.PassConfigKey.TL_CUDA_COMPILER: "nvrtc"}


@tilelang.testing.requires_cuda
def test_nvrtc_ffi_host_preparation_export_and_stream(nvrtc_ffi, tmp_path):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), n: T.int32, a: T.float32, b: T.float32, repeats: T.int32):
        if a == b:
            for step in T.serial(repeats):
                p = T.bind(n * 3 + step + 1)
                q = T.bind(p * 2 + 5)
                with T.Kernel(1, threads=128):
                    i = T.get_thread_binding()
                    B[i] = B[i] + A[i] + p + q

    context = create_backend_context("cuda", "c", "tvm_ffi")
    original = tvm.IRModule({"main": main})
    with tvm.transform.PassContext(config=nvrtc_ffi), context.target:
        mod = tvm.transform.Sequential(
            [
                tvm.tirx.transform.BindTarget(context.target),
                tilelang.transform.MaterializeKernelLaunch(),
                tilelang.transform.LowerOpaqueBlock(),
                tilelang.transform.AnnotateDeviceRegions(),
                tilelang.transform.SplitHostDevice(),
                tvm.tirx.transform.AnnotateEntryFunc(),
                tilelang.transform.MakePackedAPI(),
                tilelang.transform.LowerDeviceKernelLaunch(),
            ]
        )(original)
        host_ir = tvm.tirx.transform.Filter(get_host_call())(mod)
        device_ir = tvm.tirx.transform.Filter(get_device_call())(mod)
        device_module = device_codegen(device_ir, context)
        runtime_module = host_codegen(host_ir, context)
        runtime_module.import_module(device_module)

    library = tmp_path / ("prepared.dll" if sys.platform == "win32" else "prepared.so")
    from tilelang.jit.adapter.tvm_ffi import COMPILE_ARGS

    runtime_module.export_library(str(library), **COMPILE_ARGS)
    loaded_module = tvm.runtime.load_module(str(library))
    stream = torch.cuda.Stream()
    for module in (runtime_module, loaded_module):
        adapter = TVMFFIKernelAdapter(
            params=extrac_params(main),
            result_idx=[],
            target=context.target,
            func_or_mod=original,
            rt_mod=module,
        )
        for n, b, repeats in ((2, 1.0 + 2**-25, 3), (-4, 2.0, 2), (7, 1.0, 0), (-3, 1.0, 5)):
            with torch.cuda.stream(stream):
                inputs = torch.arange(128, dtype=torch.int32, device="cuda")
                outputs = torch.full_like(inputs, -17)
                adapter.func(inputs, outputs, n, 1.0, b, repeats)
                expected = torch.full_like(inputs, -17)
                if torch.tensor(b, dtype=torch.float32).item() == 1.0:
                    expected += repeats * (inputs + 9 * n + 8) + 3 * repeats * (repeats - 1) // 2
            stream.synchronize()
            if not torch.equal(outputs, expected):
                pytest.fail(f"Host preparation produced incorrect output for n={n}, b={b}, repeats={repeats}")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("fast_math", [False, True])
def test_nvrtc_ffi_public_compile_and_cache(nvrtc_ffi, fast_math):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32"), n: T.int32):
        with T.Kernel(1, threads=128):
            i = T.get_thread_binding()
            B[i] = A[i] * T.float32(n) + T.float32(0.5)

    from tilelang.contrib import nvrtc
    from tilelang.cuda.backend import tilelang_callback_cuda_compile

    configs = {**nvrtc_ffi, tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: fast_math}
    kernel = tilelang.compile(main, execution_backend="tvm_ffi", pass_configs=configs, compile_flags=["-lineinfo"])
    stream = torch.cuda.Stream()
    for n in (2, -3, 0, 7):
        with torch.cuda.stream(stream):
            inputs = torch.arange(128, dtype=torch.float32, device="cuda")
            outputs = torch.empty_like(inputs)
            kernel(inputs, outputs, n)
            expected = inputs * n + 0.5
        stream.synchronize()
        if not torch.equal(outputs, expected):
            pytest.fail(f"NVRTC compiled kernel produced incorrect output for n={n}")

    source = kernel.get_kernel_source()
    compile_configs = {**configs, tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-lineinfo"]}
    binary = tilelang_callback_cuda_compile(source, kernel.target, compile_configs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nvrtc, "compile_cuda", lambda *args, **kwargs: pytest.fail("CUDA binary cache missed"))
        cached = tilelang_callback_cuda_compile(source, kernel.target, compile_configs)
    if bytes(cached) != bytes(binary):
        pytest.fail("CUDA binary cache changed the compiled device bytes")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_nvrtc_ffi_tma_launch(nvrtc_ffi):
    @T.prim_func
    def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((16, 128), "float32")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 128), "float32")
            barrier = T.alloc_barrier(32)
            T.tma_copy(A, shared, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            T.copy(shared, B)

    configs = {**nvrtc_ffi, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    kernel = tilelang.compile(main, execution_backend="tvm_ffi", pass_configs=configs)
    inputs = torch.arange(16 * 128, device="cuda", dtype=torch.float32).reshape(16, 128)
    for offset in (0, 19, -7):
        source = inputs + offset
        outputs = torch.empty_like(source)
        kernel(source, outputs)
        if not torch.equal(outputs, source):
            pytest.fail(f"NVRTC TMA copy produced incorrect output for offset={offset}")


if __name__ == "__main__":
    tilelang.testing.main()
