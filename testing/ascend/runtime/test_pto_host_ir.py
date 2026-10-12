"""On-device C/MIR parity for the shared AscendC/PTO Host IR execution path."""

import pytest
import torch
import tvm_ffi

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.backend import create_backend_context
from tilelang.cache import _resolve_cache_dispatch
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter
import tilelang.testing


def test_pto_context_uses_shared_host_and_cache():
    context = create_backend_context("pto", "mir", "auto")
    assert context.module.name == "pto"
    assert context.execution_backend.name == "tvm_ffi"
    assert context.execution_backend.supports_callee_allocated_outputs
    cache, resolved, _ = _resolve_cache_dispatch("pto", "c", "tvm_ffi", False)
    assert resolved.module.name == "pto"
    assert cache.host_kernel_path.endswith(".c")


def _loop_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), steps: T.int32):
        with T.Kernel(T.ceildiv(n, 128)) as bx, T.SimtVF(threads=128):
            for i in T.Parallel(128):
                if bx * 128 + i < n:
                    B[bx * 128 + i] = A[bx * 128 + i]
        for _ in T.serial(steps):
            with T.Kernel(T.ceildiv(n, 128)) as bx, T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    if bx * 128 + i < n:
                        B[bx * 128 + i] += 1.0

    return main


def test_pto_device_module_build_without_toolchain():
    name = "tilelang_callback_pto_compile"
    original = tvm_ffi.get_global_func(name)
    entries = []

    def compile_entry(code, symbol, target, config):
        entries.append(str(symbol))
        assert "@pto.jit" in code
        assert "pto" in target.keys
        # The native module stores bytes lazily; this test never launches them.
        return bytearray(b"test-device-binary")

    try:
        tvm_ffi.register_global_func(name, compile_entry, override=True)
        artifact = tilelang.lower(_loop_kernel(), target="pto", target_host="c", enable_device_compile=True)
        assert len(entries) == 2
        assert len(set(entries)) == 2
        assert "@pto.jit" in artifact.kernel_source
        assert "for" in artifact.host_mod.script()
    finally:
        tvm_ffi.register_global_func(name, original, override=True)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("host", ["c", "mir"])
def test_host_loop_dynamic_output_stream_and_export(target, host, tmp_path):
    if host == "mir" and tvm.ffi.get_global_func("target.build.tilelang_mir", allow_missing=True) is None:
        pytest.skip("MIR host compiler is not built")
    kernel = tilelang.compile(_loop_kernel(), target=target, target_host=host, execution_backend="tvm_ffi", out_idx=[1])
    assert isinstance(kernel.adapter, TVMFFIKernelAdapter)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for size, steps in [(257, 0), (511, 3)]:
            a = torch.randn(size, device="npu", dtype=torch.float32)
            b = kernel(a, steps)
            torch.testing.assert_close(b, a + steps)

    # Exercise serialization of every imported per-entry device module. Reload
    # without regenerating PTODSL, rebuilding the device, or a Cython extension.
    path = str(tmp_path / ("kernel.tlm" if host == "mir" else "kernel.so"))
    kernel.export_library(path)
    loaded = tvm.runtime.load_module(path)
    kernel.adapter.executable = loaded
    kernel.adapter.__dict__.pop("_entry", None)
    with torch.npu.stream(stream):
        a = torch.randn(257, device="npu", dtype=torch.float32)
        torch.testing.assert_close(kernel(a, 2), a + 2)
    stream.synchronize()
