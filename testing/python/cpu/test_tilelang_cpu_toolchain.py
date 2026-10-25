"""CPU compile-flag policy consumed by the shared execution adapters."""

import sys

import pytest

from tilelang.cpu import toolchain
from tilelang import tvm
from tilelang.env import TILELANG_TEMPLATE_PATH


@pytest.mark.parametrize("backend", ["cython", "tvm_ffi"])
@pytest.mark.parametrize(
    "target,pass_configs,expected",
    [
        ("c", None, []),
        ("c", {}, []),
        ("c", {"tl.disable_vectorize_256": True}, []),
        ("c", {"tl.cpu_parallel": False}, []),
        ("c", {"tl.cpu_parallel": True}, ["-O2", "-fopenmp"]),
        ("llvm", {"tl.cpu_parallel": True}, []),
        ("cuda", {"tl.cpu_parallel": True}, []),
    ],
)
def test_cpu_compile_flag_policy(monkeypatch, backend, target, pass_configs, expected):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target)
    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    assert toolchain.get_compile_flags(target, pass_configs, execution_backend=backend) == expected


def test_cpu_openmp_runtime_flags():
    flags = toolchain.get_compile_flags(tvm.target.Target("c"), {"tl.cpu_parallel": True}, execution_backend="cython")
    assert "-O2" in flags
    if sys.platform != "win32" and (sys.platform != "darwin" or toolchain._find_libomp() is not None):
        assert "-fopenmp" in flags


@pytest.mark.parametrize("backend,expected", [("cython", ["-O2"]), ("tvm_ffi", [])])
def test_cpu_windows_serial_fallback(monkeypatch, backend, expected):
    target = tvm.target.Target("c")
    monkeypatch.setattr(toolchain.sys, "platform", "win32")
    with pytest.warns(UserWarning, match="compiling serially"):
        assert toolchain.get_compile_flags(target, {"tl.cpu_parallel": True}, execution_backend=backend) == expected


def test_tvm_ffi_uses_cpu_toolchain_flags(monkeypatch):
    from tilelang.jit.adapter import tvm_ffi

    seen = {}

    class Executable:
        def __init__(self, mod):
            assert mod is adapter.rt_mod

        def jit(self, **kwargs):
            seen.update(kwargs)

    def flags(target, pass_configs, *, execution_backend):
        assert target.same_as(adapter.target)
        assert pass_configs == {"tl.cpu_parallel": True}
        assert execution_backend == "tvm_ffi"
        return ["-O2", "-fopenmp"]

    adapter = tvm_ffi.TVMFFIKernelAdapter.__new__(tvm_ffi.TVMFFIKernelAdapter)
    adapter.rt_mod = object()
    adapter.target = tvm.target.Target("c")
    adapter.pass_configs = {"tl.cpu_parallel": True}
    adapter.compile_flags = None
    monkeypatch.setattr(tvm_ffi.runtime, "Executable", Executable)
    monkeypatch.setattr(tvm_ffi, "COMPILE_ARGS", {"options": ["-existing"]})
    monkeypatch.setattr(toolchain, "get_compile_flags", flags)
    adapter._make_executable()
    language_flags = ["/TP", "/std:c++17", "/EHsc"] if sys.platform == "win32" else ["-std=c++17"]
    assert seen["options"] == [
        "-existing",
        "-O2",
        "-fopenmp",
        "-I" + TILELANG_TEMPLATE_PATH,
        *language_flags,
    ]


def test_cpu_ffi_compile_export_and_disk_cache(monkeypatch, tmp_path):
    import torch
    import tilelang
    import tilelang.cpu.language as T
    from tilelang.cache import _dispatch_map
    from tilelang.env import env
    from tilelang.jit.kernel import JITKernel

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(env, "is_cache_enabled", lambda: True)
    cache = _dispatch_map["tvm_ffi"]
    monkeypatch.setattr(cache, "_memory_cache", {})

    @T.prim_func
    def fill(B: T.Tensor((4,), "float32")):
        with T.Kernel(1, prelude="#ifndef CPU_CUSTOM_FLAG\n#error missing CPU compile flag\n#endif\n"):
            for i in T.grid(4):
                B[i] = T.float32(3.0)

    kwargs = dict(target="c", target_host="c", out_idx=[0], compile_flags=["-DCPU_CUSTOM_FLAG=1"])
    kernel = tilelang.compile(fill, **kwargs)
    assert kernel.execution_backend == "tvm_ffi"
    assert kernel.target_host.kind.name == "c"
    torch.testing.assert_close(kernel(), torch.full((4,), 3.0))
    export_path = tmp_path / "export.so"
    kernel.export_library(str(export_path))
    exported = tvm.runtime.load_module(str(export_path))
    output = torch.empty(4)
    exported(output)
    torch.testing.assert_close(output, torch.full((4,), 3.0))
    assert list((tmp_path / "cache").rglob("executable.so"))

    cache._memory_cache.clear()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("The warm CPU invocation must load the persisted FFI library.")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", unexpected_compile)
    loaded = tilelang.compile(fill, **kwargs)
    torch.testing.assert_close(loaded(), torch.full((4,), 3.0))
