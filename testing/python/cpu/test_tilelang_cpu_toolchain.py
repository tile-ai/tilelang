"""CPU compile-flag policy consumed by the shared execution adapters."""

import sys

import pytest

from tilelang.cpu import toolchain
from tilelang import tvm


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
    monkeypatch.setattr(tvm_ffi.runtime, "Executable", Executable)
    monkeypatch.setattr(tvm_ffi, "COMPILE_ARGS", {"options": ["-existing"]})
    monkeypatch.setattr(toolchain, "get_compile_flags", flags)
    adapter._make_executable()
    assert seen["options"] == ["-existing", "-O2", "-fopenmp"]
