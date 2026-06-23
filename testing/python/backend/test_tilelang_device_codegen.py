from __future__ import annotations

import pytest

from tvm.target import Target

from tilelang.backend.device_codegen import (
    DeviceCodegen,
    allowed_device_codegens_for_target,
    register_device_codegen,
    resolve_device_codegen,
)


def _cutedsl_target() -> Target:
    target = Target("cuda")
    target_config = dict(target.export())
    target_config["keys"] = list(dict.fromkeys([*target_config.get("keys", ()), "cutedsl"]))
    return Target(target_config)


def test_cuda_device_codegen_resolves_target_variants():
    cuda_target = Target("cuda")
    cutedsl_target = _cutedsl_target()

    assert resolve_device_codegen(cuda_target).name == "cuda"
    assert resolve_device_codegen(cutedsl_target).name == "cutedsl"
    assert allowed_device_codegens_for_target(cutedsl_target) == ["cutedsl"]


def test_device_codegen_invokes_registered_global_func(monkeypatch):
    from tilelang.backend import device_codegen as registry

    calls: list[tuple[str, object, Target]] = []

    def fake_get_global_func(name: str):
        def build(mod, target):
            calls.append((name, mod, target))
            return f"built:{name}"

        return build

    monkeypatch.setattr(registry.tvm.ffi, "get_global_func", fake_get_global_func)

    cuda_target = Target("cuda")
    cutedsl_target = _cutedsl_target()
    llvm_target = Target("llvm")

    assert resolve_device_codegen(cuda_target).lower("mod", cuda_target, compile_device=False) == (
        "built:target.build.tilelang_cuda_without_compile"
    )
    assert resolve_device_codegen(cutedsl_target).lower("mod", cutedsl_target, compile_device=True) == (
        "built:target.build.tilelang_cutedsl"
    )
    assert resolve_device_codegen(llvm_target).lower("mod", llvm_target, compile_device=True) == "built:target.build.llvm"
    assert calls[0][0] == "target.build.tilelang_cuda_without_compile"
    assert calls[1][0] == "target.build.tilelang_cutedsl"
    assert calls[2][0] == "target.build.llvm"


def test_c_device_codegen_preserves_compile_unsupported_behavior():
    target = Target("c")

    with pytest.raises(ValueError, match="does not support lowering with compilation"):
        resolve_device_codegen(target).lower("mod", target, compile_device=True)


def test_device_codegen_registry_is_extensible():
    from tilelang.backend import device_codegen as registry

    target_kind = "llvm"
    old_codegen_specs = registry._DEVICE_CODEGENS.get(target_kind)
    was_loaded = target_kind in registry._LOADED_DEVICE_CODEGENS
    try:
        registry._DEVICE_CODEGENS[target_kind] = []
        registry._LOADED_DEVICE_CODEGENS.add(target_kind)
        register_device_codegen(
            target_kind,
            DeviceCodegen("unit", build_without_compile=lambda mod, target: mod),
            override=True,
        )

        target = Target(target_kind)
        assert allowed_device_codegens_for_target(target) == ["unit"]
        assert resolve_device_codegen(target).name == "unit"
    finally:
        if old_codegen_specs is None:
            registry._DEVICE_CODEGENS.pop(target_kind, None)
        else:
            registry._DEVICE_CODEGENS[target_kind] = old_codegen_specs
        if not was_loaded:
            registry._LOADED_DEVICE_CODEGENS.discard(target_kind)
