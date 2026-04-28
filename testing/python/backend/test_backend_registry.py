from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tilelang  # noqa: F401
from tvm.target import Target

from tilelang.backend.execution import get_execution_spec, get_kernel_cache, get_library_compile_spec
from tilelang.backend.errors import BackendResolutionError
from tilelang.backend.nvidia.passes import NvidiaPassHooks
from tilelang.backend.registry import (
    allowed_execution_backends_for_target,
    get_device_backend,
    resolve_execution_backend,
)
from tilelang.engine import phase


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_device_backend_registry_resolves_current_targets():
    cases = [
        ("cuda", "nvidia", "cuda", "target.build.tilelang_cuda_without_compile"),
        ("cuda -keys=cutedsl", "cutedsl", "cutedsl_py", "target.build.tilelang_cutedsl_without_compile"),
        ("hip", "amd", "hip", "target.build.tilelang_hip_without_compile"),
        ("metal", "apple", "metal", "target.build.metal"),
        ("c", "cpu-c", "c", "target.build.tilelang_c"),
        ("llvm", "cpu-llvm", "llvm", "target.build.llvm"),
        ("webgpu", "webgpu", "webgpu", "target.build.webgpu"),
    ]

    for target_spec, backend_name, source_kind, source_symbol in cases:
        backend = get_device_backend(Target(target_spec))
        assert backend.name == backend_name
        assert backend.source_kind == source_kind
        assert backend.source_builder is not None
        assert backend.source_builder.ffi_symbol == source_symbol


def test_execution_backend_compatibility_matrix_is_backend_owned():
    assert allowed_execution_backends_for_target(Target("cuda")) == ["tvm_ffi", "nvrtc", "cython"]
    assert allowed_execution_backends_for_target(Target("cuda -keys=cutedsl")) == ["cutedsl"]
    assert allowed_execution_backends_for_target(Target("hip")) == ["tvm_ffi", "cython"]
    assert allowed_execution_backends_for_target(Target("metal")) == ["tvm_ffi", "torch"]
    assert allowed_execution_backends_for_target(Target("c")) == ["cython", "tvm_ffi"]
    assert allowed_execution_backends_for_target(Target("llvm")) == ["cython", "tvm_ffi"]
    assert allowed_execution_backends_for_target(Target("webgpu")) == ["cython", "tvm_ffi"]


def test_execution_backend_resolution_preserves_defaults_and_aliases():
    assert resolve_execution_backend("auto", Target("cuda")) == "tvm_ffi"
    assert resolve_execution_backend(None, Target("hip")) == "tvm_ffi"
    assert resolve_execution_backend("auto", Target("cuda -keys=cutedsl")) == "cutedsl"
    assert resolve_execution_backend("auto", Target("c")) == "cython"
    assert resolve_execution_backend("dlpack", Target("cuda")) == "tvm_ffi"


def test_execution_backend_resolution_rejects_invalid_pairs():
    with pytest.raises(ValueError, match="Invalid execution backend"):
        resolve_execution_backend("torch", Target("cuda"))


def test_unknown_target_reports_backend_resolution_error():
    with pytest.raises(BackendResolutionError, match="No TileLang device backend"):
        get_device_backend(Target("vulkan"))


def test_execution_specs_own_adapter_codegen_and_cache_metadata():
    from tilelang.cache import _dispatch_map

    cuda_backend = get_device_backend(Target("cuda"))
    cutedsl_backend = get_device_backend(Target("cuda -keys=cutedsl"))

    tvm_ffi = get_execution_spec(Target("cuda"), "tvm_ffi")
    assert tvm_ffi.requires_host_codegen is True
    assert tvm_ffi.requires_device_compile is True
    assert tvm_ffi.cache_artifact.kernel_lib_path == "executable.so"
    assert get_kernel_cache(Target("cuda"), "tvm_ffi").kernel_lib_path == "executable.so"
    assert _dispatch_map["tvm_ffi"].kernel_lib_path == "executable.so"

    nvrtc = cuda_backend.execution_spec("nvrtc")
    assert nvrtc.requires_host_codegen is False
    assert nvrtc.requires_device_compile is False
    assert nvrtc.python_source_wrapper_factory is not None
    assert nvrtc.cache_artifact.kernel_lib_path == "kernel.cubin"
    assert "kernel.py" in nvrtc.cache_artifact.extra_required_paths
    assert nvrtc.cache_factory().kernel_lib_path == "kernel.cubin"
    assert _dispatch_map["nvrtc"].kernel_lib_path == "kernel.cubin"

    cutedsl = cutedsl_backend.execution_spec("cutedsl")
    assert cutedsl.python_source_wrapper_factory is not None
    assert cutedsl.cache_artifact.kernel_lib_path == "kernel.py"
    assert cutedsl.cache_artifact.device_kernel_path == "kernel.py"
    assert "launcher_lib.so" in cutedsl.cache_artifact.extra_required_paths
    assert _dispatch_map["cutedsl"].kernel_lib_path == "kernel.py"
    assert sorted(_dispatch_map) == ["cutedsl", "cython", "nvrtc", "torch", "tvm_ffi"]

    assert get_execution_spec(Target("hip"), "cython").c_source_wrapper_factory is not None
    assert get_execution_spec(Target("c"), "cython").c_source_wrapper_factory is not None
    assert get_library_compile_spec(Target("cuda")).source_suffix == ".cu"
    assert get_library_compile_spec(Target("hip")).source_suffix == ".cpp"
    assert get_library_compile_spec(Target("c")).source_suffix == ".cpp"


def test_backend_pass_hooks_are_wired_into_shared_pipeline():
    lower_source = inspect.getsource(phase.LowerAndLegalize)
    optimize_source = inspect.getsource(phase.OptimizeForTarget)
    nvidia_hook_source = inspect.getsource(NvidiaPassHooks)

    assert "backend.pass_hooks.pre_layout" in lower_source
    assert lower_source.index("backend.pass_hooks.pre_layout") < lower_source.index("PipelinePlanning")
    assert "backend.pass_hooks.post_tile_lowering" in lower_source
    assert "backend.pass_hooks.before_split_host_device" in optimize_source
    assert "backend.pass_hooks.after_split_host_device" in optimize_source
    assert "backend.pass_hooks.before_device_codegen" in optimize_source

    assert "ProducerConsumerWarpSpecialized" in nvidia_hook_source
    assert "LowerBlackwell2SM" in nvidia_hook_source
    assert "MarkCudaSyncCalls" in nvidia_hook_source
    assert "PersistThreadblock" in nvidia_hook_source


def test_native_backend_cmake_ownership_points_are_staged():
    top_level_cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    nvidia_cmake = (REPO_ROOT / "src/backend/nvidia/CMakeLists.txt").read_text()
    amd_cmake = (REPO_ROOT / "src/backend/amd/CMakeLists.txt").read_text()
    common_cmake = (REPO_ROOT / "src/backend/common/CMakeLists.txt").read_text()

    assert "tilelang_configure_nvidia_backend" in top_level_cmake
    assert "tilelang_configure_amd_backend" in top_level_cmake
    assert "elseif(USE_ROCM)" not in top_level_cmake
    assert "elseif(USE_CUDA)" not in top_level_cmake
    assert "src/backend/nvidia/runtime/runtime.cc" in nvidia_cmake
    assert "src/target/codegen_cuda.cc" in nvidia_cmake
    assert "src/target/stubs/cuda.cc" in nvidia_cmake
    assert "src/target/codegen_hip.cc" in amd_cmake
    assert "src/target/stubs/hip.cc" in amd_cmake
    assert "lower_hopper_intrin.cc" in common_cmake
    assert "list(REMOVE_ITEM _common_srcs" in common_cmake


def test_backend_imports_do_not_eagerly_require_optional_sdk_adapters():
    import importlib

    for module_name in [
        "tilelang",
        "tilelang.backend",
        "tilelang.cache",
        "tilelang.jit.execution_backend",
        "tilelang.jit.kernel",
    ]:
        importlib.import_module(module_name)
