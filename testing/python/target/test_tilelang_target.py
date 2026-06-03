from __future__ import annotations

from pathlib import Path

import pytest

import tilelang
import tilelang.backend.target as target_registry
from tilelang.backend import (
    Target,
    TargetKind,
    list_target_kinds,
    list_target_presets,
    register_target_kind,
    register_target_preset,
    resolve_target_execution_backend,
)


def test_backend_package_does_not_own_backend_specific_target_presets():
    backend_init = Path(target_registry.__file__).with_name("__init__.py").read_text()

    assert "cuda" not in backend_init
    assert "hip" not in backend_init
    assert "h100" not in backend_init
    assert "mi300x" not in backend_init
    assert "cutedsl" not in backend_init


def test_tilelang_exports_target():
    assert tilelang.Target is Target


def test_target_owns_basic_target_parsing():
    target = Target("cuda", arch="sm_90", execution_backend="dlpack")

    tvm_target_input, execution_backend = resolve_target_execution_backend(target, None)

    assert tvm_target_input == {"kind": "cuda", "arch": "sm_90"}
    assert execution_backend == "tvm_ffi"

    parsed = Target("cuda -arch=sm_90 -keys=cuda,gpu", host="llvm")
    assert parsed.to_config() == {
        "kind": "cuda",
        "arch": "sm_90",
        "keys": ["cuda", "gpu"],
        "host": "llvm",
    }


def test_backend_owned_target_kinds_and_presets_normalize_to_tvm_target_input():
    assert "cuda" in list_target_kinds()
    assert "hip" in list_target_kinds()
    assert "cpu" in list_target_kinds()
    assert "h100" in list_target_presets()
    assert "mi300x" in list_target_presets()

    assert Target("h100").to_tvm_target_input() == {"kind": "cuda", "arch": "sm_90a"}
    assert Target("mi300x").to_tvm_target_input() == {"kind": "hip", "mcpu": "gfx942"}
    assert Target("cpu").to_tvm_target_input() == {"kind": "llvm"}


def test_cutedsl_is_cuda_owned_target_kind_variant():
    assert Target("cutedsl").to_tvm_target_input() == {"kind": "cuda", "keys": ["cutedsl"]}
    assert Target("cutedsl", arch="sm_90").to_tvm_target_input() == {
        "kind": "cuda",
        "arch": "sm_90",
        "keys": ["cutedsl"],
    }


def test_target_kind_registration_does_not_require_tvm_target_kind():
    name = "unit-accelerator"

    def normalize(spec):
        return {"kind": "llvm", "keys": [spec.kind], **dict(spec.attrs)}

    old_kind = target_registry._TARGET_KINDS.get(name)
    try:
        register_target_kind(TargetKind(name, normalize=normalize), override=True)

        target = Target(name, mcpu="native")
        assert target.to_tvm_target_input() == {"kind": "llvm", "keys": [name], "mcpu": "native"}
        assert target.to_tvm_target().kind.name == "llvm"
    finally:
        if old_kind is None:
            target_registry._TARGET_KINDS.pop(name, None)
        else:
            target_registry._TARGET_KINDS[name] = old_kind


def test_target_preset_registration_is_extensible():
    name = "unit-lite-target"

    def resolve(spec):
        attrs = dict(spec.attrs)
        return {"kind": "llvm", **attrs}

    old_preset = target_registry._TARGET_PRESETS.get(name)
    try:
        register_target_preset(name, resolve, override=True)

        assert Target(name, mcpu="native").to_tvm_target_input() == {"kind": "llvm", "mcpu": "native"}
    finally:
        if old_preset is None:
            target_registry._TARGET_PRESETS.pop(name, None)
        else:
            target_registry._TARGET_PRESETS[name] = old_preset


def test_auto_target_uses_registered_driver_detectors():
    name = "unit-auto-target"

    def detect():
        return Target(name, mcpu="native")

    old_kind = target_registry._TARGET_KINDS.get(name)
    try:
        register_target_kind(name, tvm_kind="llvm", detect=detect, priority=10000, override=True)

        assert Target("auto").to_tvm_target_input() == {"kind": "llvm", "mcpu": "native"}
    finally:
        if old_kind is None:
            target_registry._TARGET_KINDS.pop(name, None)
        else:
            target_registry._TARGET_KINDS[name] = old_kind


def test_target_execution_backend_conflict_reports_both_sources():
    target = Target("llvm", execution_backend="cython")

    with pytest.raises(ValueError, match="Conflicting execution backend"):
        resolve_target_execution_backend(target, "tvm_ffi")


def test_explicit_compile_execution_backend_overrides_target_auto():
    target = Target("llvm", execution_backend="auto")

    tvm_target_input, execution_backend = resolve_target_execution_backend(target, "tvm_ffi")

    assert tvm_target_input == "llvm"
    assert execution_backend == "tvm_ffi"
