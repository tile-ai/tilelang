"""Native compiler process helpers, provider libraries, and persistent artifacts."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.support import tar, utils

from tilelang.contrib import cc, hipcc, msvc, nvcc


def test_toolchains_import_without_legacy_tvm():
    assert "tvm" not in sys.modules
    assert cc._utils is utils
    assert cc._tar is tar
    assert nvcc.Target is tvm_tirx.target.Target


def test_native_cc_compiles_and_executes(tmp_path):
    if not cc.get_cc():
        pytest.skip("C++ compiler is unavailable")
    source = tmp_path / "kernel.cc"
    binary = tmp_path / ("kernel.dll" if sys.platform == "win32" else "kernel.so")
    source.write_text('extern "C" int triple_plus_one(int x) { return x * 3 + 1; }\n')
    cc.create_shared(str(binary), [str(source)])
    kernel = ctypes.CDLL(str(binary)).triple_plus_one
    kernel.argtypes = [ctypes.c_int]
    kernel.restype = ctypes.c_int
    assert [kernel(x) for x in (-11, 0, 7)] == [-32, 1, 22]


@pytest.mark.skipif(sys.platform == "win32", reason="archive toolchain is Unix-only")
def test_native_tar_archive_input(tmp_path):
    compiler = cc.get_cc()
    if not compiler or not shutil.which("ar"):
        pytest.skip("C++ compiler/ar is unavailable")
    source = tmp_path / "part.cc"
    obj = tmp_path / "part.o"
    source.write_text('extern "C" int part(int x) { return x * 5; }\n')
    subprocess.run([compiler, "-fPIC", "-c", str(source), "-o", str(obj)], check=True, capture_output=True)
    packed = tmp_path / "part.tar"
    tar.tar(str(packed), [str(obj)])
    archive = tmp_path / "parts.a"
    cc.create_staticlib(str(archive), [str(packed)])
    caller = tmp_path / "caller.cc"
    caller.write_text('extern "C" int part(int); extern "C" int answer(int x) { return part(x) + 2; }\n')
    binary = tmp_path / "caller.so"
    cc.create_shared(str(binary), [str(caller), str(archive)])
    kernel = ctypes.CDLL(str(binary)).answer
    kernel.argtypes = [ctypes.c_int]
    kernel.restype = ctypes.c_int
    assert kernel(8) == 42


def test_nvcc_uses_native_target_and_persistent_pass_context(tmp_path):
    compiler = nvcc.get_nvcc_compiler()
    if not compiler or not shutil.which(compiler):
        pytest.skip("NVCC is unavailable")
    artifacts = tmp_path / "kernels"
    source = 'extern "C" __global__ void bump(float* p) { p[threadIdx.x] += 2.0f; }'
    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}), PassContext(config={"cuda.kernels_output_dir": str(artifacts)}):
        binary = nvcc.compile_cuda(source, target_format="ptx")
    assert b".visible .entry bump" in binary
    inputs = list(artifacts.glob("*.cu"))
    outputs = list(artifacts.glob("*.ptx"))
    assert len(inputs) == len(outputs) == 1
    assert inputs[0].read_text() == source
    assert outputs[0].read_bytes() == binary


def test_nvrtc_uses_native_target_context(tmp_path):
    pytest.importorskip("cuda.bindings.nvrtc")
    from tilelang.contrib import nvrtc

    source = 'extern "C" __global__ void bump(float* p) { p[threadIdx.x] += 2.0f; }'
    templates = Path(__file__).resolve().parents[3] / "src"
    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}):
        binary = nvrtc.compile_cuda(source, target_format="ptx", options=["--std=c++17", f"-I{templates}"], verbose=True)
    (tmp_path / "kernel.ptx").write_bytes(binary)
    assert b".visible .entry bump" in binary


def test_hip_native_pass_context_artifact_paths(tmp_path):
    artifacts = tmp_path / "hip-kernels"
    with PassContext(config={"hip.kernels_output_dir": str(artifacts)}):
        output_dir = PassContext.current().config["hip.kernels_output_dir"]
        temp = utils.tempdir()
        first = hipcc._resolve_artifact_paths(temp, "kernel", "hsaco", output_dir)
        second = hipcc._resolve_artifact_paths(temp, "kernel", "hsaco", output_dir)
    assert first != second
    assert all(Path(path).parent == artifacts for path in (*first, *second))
    assert Path(first[0]).exists() and Path(second[0]).exists()


def test_windows_import_libraries_follow_selected_native_provider(tmp_path, monkeypatch):
    # This validates selection on every platform. Actual Windows linking needs
    # a Windows runner; these files are not represented as loadable DLLs.
    provider = tmp_path / "native-provider"
    provider.mkdir()
    runtime = provider / "tvm_runtime.dll"
    implib = runtime.with_suffix(".lib")
    ffi = tmp_path / "ffi" / "tvm_ffi.lib"
    ffi.parent.mkdir()
    for path in (runtime, implib, ffi, provider / "tvm.lib"):
        path.touch()
    monkeypatch.setattr(tvm_tirx.libinfo, "find_libtvm_runtime", lambda: str(runtime))
    monkeypatch.setattr(tvm_ffi.libinfo, "find_windows_implib", lambda: str(ffi))
    directories, libraries = msvc._find_import_libs()
    assert libraries == [str(implib), str(ffi)]
    assert directories == [str(provider), str(ffi.parent)]


def test_windows_missing_runtime_import_lib_cannot_fall_back_to_legacy(tmp_path, monkeypatch):
    runtime = tmp_path / "tvm_runtime.dll"
    runtime.touch()
    (tmp_path / "tvm.lib").touch()
    monkeypatch.setattr(tvm_tirx.libinfo, "find_libtvm_runtime", lambda: str(runtime))
    with pytest.raises(RuntimeError, match="Missing TIRx runtime import library"):
        msvc._find_import_libs()
