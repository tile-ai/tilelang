"""HIP SDK discovery and compiler environment regression tests without a GPU."""

import os
from types import SimpleNamespace

import pytest

from tilelang.contrib import msvc, rocm
from tilelang import _rocm_sdk


def _make_sdk(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "include/hip").mkdir(parents=True)
    (tmp_path / "include/hip/hip_runtime.h").touch()
    (tmp_path / "bin/hipcc.exe").touch()
    return tmp_path


@pytest.mark.parametrize("variable", ["ROCM_PATH", "ROCM_HOME", "HIP_PATH"])
def test_windows_hipcc_explicit_sdk(tmp_path, monkeypatch, variable):
    sdk = _make_sdk(tmp_path)
    for name in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rocm.sys, "platform", "win32")
    monkeypatch.setenv(variable, str(sdk))
    assert rocm.find_hipcc() == str(sdk / "bin/hipcc.exe")
    assert rocm.find_rocm_path() == str(sdk)


def test_pip_sdk_discovery_without_environment_or_sdk_import(tmp_path, monkeypatch):
    sdk = _make_sdk(tmp_path)
    for name in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    compiler = SimpleNamespace(name="hipcc.exe", locate=lambda: sdk / "bin/hipcc.exe")
    monkeypatch.setattr(_rocm_sdk.importlib.metadata, "files", lambda package: [compiler] if package == "rocm-sdk-core" else [])
    assert _rocm_sdk.find_rocm_home() == str(sdk)


def test_windows_hip_arch_uses_amdgpu_arch(tmp_path, monkeypatch):
    tool = tmp_path / "lib/llvm/bin/amdgpu-arch.exe"
    tool.parent.mkdir(parents=True)
    tool.touch()
    monkeypatch.setattr(rocm.sys, "platform", "win32")

    def query(command, **kwargs):
        assert command == [str(tool)]
        return b"gfx1103\n"

    monkeypatch.setattr(rocm.subprocess, "check_output", query)
    assert rocm.get_rocm_arch(str(tmp_path)) == "gfx1103"


def test_windows_hip_arch_does_not_silently_use_gfx900(tmp_path, monkeypatch):
    monkeypatch.setattr(rocm.sys, "platform", "win32")
    monkeypatch.setattr(rocm.subprocess, "check_output", lambda *args, **kwargs: b"")
    with pytest.raises(RuntimeError, match="No AMD GPU architecture"):
        rocm.get_rocm_arch(str(tmp_path))


def test_windows_hip_environment_preserves_parent(tmp_path, monkeypatch):
    sdk = _make_sdk(tmp_path)
    bitcode = sdk / "lib/llvm/amdgcn/bitcode"
    bitcode.mkdir(parents=True)
    (bitcode / "ocml.bc").touch()
    original = {"Path": "host-tools", "INCLUDE": "host-headers"}
    monkeypatch.setattr(rocm.sys, "platform", "win32")
    monkeypatch.setattr(rocm, "find_rocm_path", lambda: str(sdk))
    monkeypatch.setattr(msvc, "get_msvc_subprocess_env", lambda: original)
    compiler_env = rocm.get_hipcc_subprocess_env()
    assert compiler_env["HIP_PATH"] == str(sdk)
    assert compiler_env["HIP_DEVICE_LIB_PATH"] == str(bitcode)
    assert compiler_env["PATH"] == str(sdk / "bin") + os.pathsep + "host-tools"
    assert "Path" not in compiler_env
    assert original == {"Path": "host-tools", "INCLUDE": "host-headers"}


def test_vsdevcmd_environment_has_one_path_key(monkeypatch):
    # A normal dict containing Path and PATH creates a duplicate key in the
    # Windows environment block and can hide the compiler's SDK DLL directory.
    original = {"Path": "parent-tools"}
    monkeypatch.setattr(msvc.os, "environ", original)
    monkeypatch.setattr(
        msvc.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="PATH=vc-tools\nINCLUDE=vc-headers\n")
    )
    from tilelang import _host_toolchain

    compiler_env = _host_toolchain._import_vsdevcmd_environment("VsDevCmd.bat")
    assert compiler_env == {"PATH": "vc-tools", "INCLUDE": "vc-headers"}
    assert original == {"Path": "parent-tools"}
