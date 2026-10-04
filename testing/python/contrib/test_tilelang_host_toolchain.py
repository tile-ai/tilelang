"""Host-toolchain regressions that need no CUDA/ROCm SDK or GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(sys.platform != "win32", reason="cmd's environment encoding is Windows-specific")
def test_host_environment_preserves_unicode(tmp_path, monkeypatch):
    from tilelang._host_toolchain import _import_vsdevcmd_environment

    value = "C:\\头文件目录"
    monkeypatch.setenv("TILELANG_TEST_INCLUDE", value)
    batch = tmp_path / "environment.bat"
    batch.write_bytes(b"@echo off\r\nset INCLUDE=%TILELANG_TEST_INCLUDE%\r\n")
    env = _import_vsdevcmd_environment(str(batch))
    assert env["INCLUDE"] == value


@pytest.mark.skipif(sys.platform != "win32", reason="Native MSVC diagnostics are Windows-specific")
def test_native_msvc_reports_clang_cl_requirement(tmp_path):
    from tilelang._host_toolchain import get_msvc_subprocess_env, get_env_path

    compiler = shutil.which("cl.exe", path=get_env_path(get_msvc_subprocess_env() or {}))
    if not compiler:
        pytest.skip("Visual Studio Build Tools are unavailable")
    result = subprocess.run(
        [
            _cmake(),
            "-S",
            str(ROOT),
            "-B",
            str(tmp_path / "build"),
            "-G",
            "Ninja",
            f"-DCMAKE_C_COMPILER={compiler}",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "currently require clang-cl" in " ".join((result.stdout + result.stderr).split())


def _cmake():
    command = shutil.which("cmake")
    if not command:
        pytest.skip("cmake is unavailable")
    return command


@pytest.mark.parametrize("environment_override", [False, True])
def test_disabled_cuda_does_not_probe_sdk(tmp_path, environment_override):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "FindCUDAToolkit.cmake").write_text('message(FATAL_ERROR "Disabled CUDA was probed")\n')
    source = tmp_path / "probe.cmake"
    source.write_text(
        f'set(CMAKE_MODULE_PATH "{modules.as_posix()}")\n'
        + ("" if environment_override else "set(USE_CUDA OFF)\n")
        + f'include("{(ROOT / "cmake/FindPipCUDAToolkit.cmake").as_posix()}")\n'
    )
    env = {**os.environ, "USE_CUDA": "OFF", "WITH_PIP_CUDA_TOOLCHAIN": str(tmp_path / "missing-sdk")}
    subprocess.run([_cmake(), "-P", str(source)], env=env, check=True, capture_output=True)


@pytest.mark.parametrize("missing_sdk_exit", [0, 1])
def test_sdk_probe_falls_back_to_virtualenv(tmp_path, missing_sdk_exit):
    fallback = tmp_path / "fallback env"
    venv.EnvBuilder(with_pip=False).create(fallback)
    helper = tmp_path / "probe.py"
    helper.write_text(
        "import pathlib, sys\n"
        + f"if pathlib.Path(sys.prefix) != pathlib.Path({str(fallback)!r}): sys.exit({missing_sdk_exit})\n"
        + "print(sys.argv[1])\n"
    )
    sdk = (tmp_path / "SDK with spaces").as_posix()
    source = tmp_path / "probe.cmake"
    source.write_text(
        f'include("{(ROOT / "cmake/PythonToolchain.cmake").as_posix()}")\n'
        + f'set(Python_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        + f'tilelang_probe_python_sdk("{helper.as_posix()}" sdk python "{sdk}")\n'
        + f'if(NOT sdk STREQUAL "{sdk}")\nmessage(FATAL_ERROR "SDK fallback failed: ${{sdk}}")\nendif()\n'
        + 'if(NOT python MATCHES "fallback env")\nmessage(FATAL_ERROR "Wrong SDK interpreter: ${python}")\nendif()\n'
    )
    subprocess.run([_cmake(), "-P", str(source)], env={**os.environ, "VIRTUAL_ENV": str(fallback)}, check=True, capture_output=True)


@pytest.mark.skipif(sys.platform != "win32", reason="MSVC environment persistence is Windows-specific")
@pytest.mark.parametrize("compiler_mode", ["auto", "explicit", "developer_shell"])
def test_ninja_build_restores_host_environment(tmp_path, compiler_mode):
    from tilelang._host_toolchain import get_windows_compiler, get_msvc_subprocess_env, get_env_path

    compiler, _ = get_windows_compiler()
    if compiler_mode != "auto":
        # Select cl.exe even when automatic discovery prefers clang-cl.
        compiler = shutil.which("cl.exe", path=get_env_path(get_msvc_subprocess_env() or {}))
    if not compiler:
        pytest.skip("Visual Studio Build Tools are unavailable")
    source = tmp_path / "source"
    source.mkdir()
    cmake = _cmake()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        f'include("{(ROOT / "cmake/HostToolchain.cmake").as_posix()}")\n'
        "project(host_environment C CXX)\n"
        "tilelang_enable_host_launchers()\n"
        "add_library(smoke SHARED smoke.cc)\n"
        "add_executable(probe probe.cc)\n"
        "target_link_libraries(probe PRIVATE smoke)\n"
    )
    (source / "smoke.cc").write_text(
        '#include <string>\nextern "C" __declspec(dllexport) int answer() { return std::string("hello").size(); }\n'
    )
    (source / "probe.cc").write_text('extern "C" __declspec(dllimport) int answer();\nint main() { return answer() == 5 ? 0 : 1; }\n')
    build = tmp_path / "build"
    env = os.environ.copy()
    if compiler_mode == "developer_shell":
        env = dict(get_msvc_subprocess_env())
    else:
        for key in ("INCLUDE", "LIB", "LIBPATH", "VSCMD_VER", "VCINSTALLDIR"):
            env.pop(key, None)
    configure = [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja", f"-DPython_EXECUTABLE={sys.executable}"]
    if compiler_mode == "explicit":
        configure += [f"-DCMAKE_C_COMPILER={compiler}", f"-DCMAKE_CXX_COMPILER={compiler}"]
    subprocess.run(configure, env=env, check=True, capture_output=True)
    cache_before = (build / "CMakeCache.txt").read_text()
    subprocess.run([cmake, "-S", str(source), "-B", str(build)], env=env, check=True, capture_output=True)
    cache_after = (build / "CMakeCache.txt").read_text()
    compiler_entry = next(line for line in cache_after.splitlines() if line.startswith("CMAKE_CXX_COMPILER:"))
    assert Path(compiler_entry.split("=", 1)[1]).resolve() == Path(compiler).resolve()
    # No accumulating include/link flags and no overwritten explicit compiler.
    for key in ("CMAKE_C_FLAGS:", "CMAKE_CXX_FLAGS:", "CMAKE_SHARED_LINKER_FLAGS:", "CMAKE_CXX_COMPILER:"):
        before = [line for line in cache_before.splitlines() if line.startswith(key)]
        after = [line for line in cache_after.splitlines() if line.startswith(key)]
        assert before == after
    subprocess.run([cmake, "--build", str(build)], env=env, check=True, capture_output=True)
    subprocess.run([str(build / "probe.exe")], env=env, check=True)
