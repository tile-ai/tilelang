from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.backend import (
    BackendModule,
    get_backend_module,
    get_backend_module_for_target_kind,
    list_backend_modules,
    register_backend_module,
    resolve_backend_module,
)
from tilelang.backend.device_codegen import resolve_device_codegen
from tilelang.backend.execution_backend import allowed_backends_for_target
from tilelang.backend.host_codegen import resolve_host_codegen
from tilelang.backend.pass_pipeline import resolve_pipeline


def test_builtin_backend_modules_define_one_import_boundary():
    expected = {
        "cuda": ("cuda",),
        "rocm": ("hip",),
        "cpu": ("c", "llvm"),
        "metal": ("metal",),
        "webgpu": ("webgpu",),
    }

    assert {name: module.target_kinds for name, module in list_backend_modules().items()} == expected


def test_list_backend_modules_returns_copy():
    modules = list_backend_modules()
    modules.clear()

    assert list_backend_modules()


def test_backend_module_registration_is_idempotent():
    existing = get_backend_module("cuda")

    assert register_backend_module(BackendModule("cuda", ("cuda",))) is existing


def test_backend_module_rejects_different_redeclaration():
    with pytest.raises(ValueError, match="different declaration"):
        register_backend_module(BackendModule("cuda", ("cuda", "unit-cuda")))


def test_backend_module_rejects_target_owned_by_another_module():
    with pytest.raises(ValueError, match="already owned"):
        register_backend_module(BackendModule("unit-collision", ("cuda",)))


@pytest.mark.parametrize(
    ("target_kind", "module_name", "component_name"),
    [
        ("c", "cpu", "c"),
        ("llvm", "cpu", "llvm"),
        ("cuda", "cuda", "cuda"),
        ("hip", "rocm", "hip"),
        ("metal", "metal", "metal"),
        ("webgpu", "webgpu", "webgpu"),
    ],
)
def test_builtin_module_registers_all_backend_components(target_kind: str, module_name: str, component_name: str):
    target = tvm.target.Target(target_kind)
    module = resolve_backend_module(target)

    assert module.name == module_name
    assert get_backend_module_for_target_kind(target_kind) is module
    assert module.matches(target)
    assert resolve_pipeline(target).name == target_kind
    assert resolve_device_codegen(target).name == component_name
    assert allowed_backends_for_target(target)
    if target_kind in {"c", "llvm"}:
        assert resolve_host_codegen(target).name == target_kind


def test_cuda_module_owns_compile_callbacks():
    resolve_backend_module(tvm.target.Target("cuda"))
    resolve_backend_module(tvm.target.Target("hip"))

    assert tvm.ffi.get_global_func("tilelang_callback_cuda_validate")
    assert tvm.ffi.get_global_func("tilelang_callback_cuda_compile")
    assert tvm.ffi.get_global_func("tilelang_callback_hip_compile")
