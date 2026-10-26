from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT = Path(__file__).resolve().parents[3]
EXPORT_HOOK = PROJECT / "cmake" / "tirx_extension_exports.cmake"


@pytest.mark.parametrize("integration", ["source", "standalone", "unmodified"])
def test_extension_visibility_is_owned_by_tilelang(tmp_path, integration):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to configure the native dependency fixture")
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "empty.cc").write_text("int extension_fixture() { return 0; }\n")
    (dependency / "CMakeLists.txt").write_text(
        "project(tvm_tirx LANGUAGES CXX)\n"
        "foreach(dependency_target tvm_tirx_compiler_objs tvm_runtime_objs tvm_runtime_cuda_objs tvm_ffi_objs unrelated)\n"
        "  add_library(${dependency_target} OBJECT empty.cc)\n"
        "  set_target_properties(${dependency_target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)\n"
        "endforeach()\n"
        'file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/visibility.txt CONTENT "'
        "$<TARGET_PROPERTY:tvm_tirx_compiler_objs,CXX_VISIBILITY_PRESET>|"
        "$<TARGET_PROPERTY:tvm_tirx_compiler_objs,VISIBILITY_INLINES_HIDDEN>\\n"
        "$<TARGET_PROPERTY:tvm_runtime_objs,CXX_VISIBILITY_PRESET>|"
        "$<TARGET_PROPERTY:tvm_runtime_cuda_objs,VISIBILITY_INLINES_HIDDEN>\\n"
        "$<TARGET_PROPERTY:tvm_ffi_objs,CXX_VISIBILITY_PRESET>|"
        '$<TARGET_PROPERTY:unrelated,VISIBILITY_INLINES_HIDDEN>\\n")\n'
    )
    configuration = "cmake_minimum_required(VERSION 3.26)\nproject(extension_fixture LANGUAGES CXX)\n"
    if integration == "standalone":
        configuration += f'set(CMAKE_PROJECT_tvm_tirx_INCLUDE "{EXPORT_HOOK.as_posix()}")\n'
    configuration += "add_subdirectory(dependency)\n"
    if integration == "source":
        configuration += f'include("{EXPORT_HOOK.as_posix()}")\n'
        configuration += 'tilelang_configure_tirx_exports("${CMAKE_CURRENT_SOURCE_DIR}/dependency")\n'
    (tmp_path / "CMakeLists.txt").write_text(configuration)
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    visibility = (tmp_path / "build" / "visibility.txt").read_text().splitlines()
    expected = "hidden|ON" if integration == "unmodified" else "default|OFF"
    assert visibility[:2] == [expected, expected]
    assert visibility[2] == "hidden|ON"
