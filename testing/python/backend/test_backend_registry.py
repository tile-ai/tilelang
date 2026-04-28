from __future__ import annotations

import pytest

import tilelang  # noqa: F401
from tvm.target import Target

from tilelang.backend.errors import BackendResolutionError
from tilelang.backend.registry import (
    allowed_execution_backends_for_target,
    get_device_backend,
    resolve_execution_backend,
)


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

