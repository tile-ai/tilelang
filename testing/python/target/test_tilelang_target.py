from __future__ import annotations

import pytest

import tilelang
from tvm.target import Target
import tilelang.backend.target as target_registry
from tilelang.backend.execution_backend import (
    ExecutionBackendSpec,
    allowed_backends_for_target,
    register_execution_backend,
    resolve_execution_backend,
)
from tilelang.backend.target import auto_detect_target, list_target_detectors, register_target_detector


def test_tilelang_does_not_export_target_wrapper():
    assert not hasattr(tilelang, "Target")


def test_auto_target_uses_registered_detectors():
    name = "unit-auto-target"
    old_detector = target_registry._TARGET_DETECTORS.get(name)
    try:
        register_target_detector(name, lambda: Target({"kind": "llvm", "mcpu": "native"}), priority=10000, override=True)

        target = auto_detect_target()

        assert isinstance(target, Target)
        assert target.kind.name == "llvm"
        assert str(target.attrs["mcpu"]) == "native"
        assert name in list_target_detectors()
    finally:
        if old_detector is None:
            target_registry._TARGET_DETECTORS.pop(name, None)
        else:
            target_registry._TARGET_DETECTORS[name] = old_detector


def test_auto_target_detector_falls_through_none_result():
    low_name = "unit-auto-none"
    high_name = "unit-auto-fallback"
    old_low = target_registry._TARGET_DETECTORS.get(low_name)
    old_high = target_registry._TARGET_DETECTORS.get(high_name)
    try:
        register_target_detector(low_name, lambda: None, priority=20000, override=True)
        register_target_detector(high_name, lambda: "llvm", priority=10000, override=True)

        assert auto_detect_target() == "llvm"
    finally:
        if old_low is None:
            target_registry._TARGET_DETECTORS.pop(low_name, None)
        else:
            target_registry._TARGET_DETECTORS[low_name] = old_low
        if old_high is None:
            target_registry._TARGET_DETECTORS.pop(high_name, None)
        else:
            target_registry._TARGET_DETECTORS[high_name] = old_high


def test_execution_backend_registry_resolves_target_policy():
    target_kind = "llvm"
    target = Target({"kind": target_kind})
    from tilelang.backend import execution_backend as backend_registry

    old_execution_specs = backend_registry._EXECUTION_BACKENDS.get(target_kind)
    was_loaded = target_kind in backend_registry._LOADED_EXECUTION_BACKENDS
    try:
        backend_registry._EXECUTION_BACKENDS[target_kind] = []
        backend_registry._LOADED_EXECUTION_BACKENDS.add(target_kind)
        register_execution_backend(target_kind, ExecutionBackendSpec("slow", priority=1), override=True)
        register_execution_backend(target_kind, ExecutionBackendSpec("fast", priority=10), override=True)

        assert allowed_backends_for_target(target) == ["fast", "slow"]
        assert resolve_execution_backend("auto", target) == "fast"
        assert resolve_execution_backend("slow", target) == "slow"
    finally:
        if old_execution_specs is None:
            backend_registry._EXECUTION_BACKENDS.pop(target_kind, None)
        else:
            backend_registry._EXECUTION_BACKENDS[target_kind] = old_execution_specs
        if not was_loaded:
            backend_registry._LOADED_EXECUTION_BACKENDS.discard(target_kind)


def test_execution_backend_registry_rejects_invalid_backend():
    target = Target("llvm")

    with pytest.raises(ValueError, match="Invalid execution backend"):
        resolve_execution_backend("nvrtc", target)
