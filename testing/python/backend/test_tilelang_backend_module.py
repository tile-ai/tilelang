from __future__ import annotations

from dataclasses import replace

import pytest

from tilelang import tvm
from tilelang.backend import BackendContext, create_backend_context, get_backend, list_backends, register_backend


def test_builtin_backend_modules_are_explicit():
    expected = {
        "cuda": (("cuda",), ["tvm_ffi", "nvrtc", "cython"]),
        "cutedsl": (("cuda",), ["cutedsl"]),
        "rocm": (("hip",), ["tvm_ffi", "cython"]),
        "cpu": (("c", "llvm"), ["cython", "tvm_ffi"]),
        "metal": (("metal",), ["torch", "tvm_ffi"]),
        "webgpu": (("webgpu",), ["tvm_ffi"]),
    }

    assert {
        name: (backend.target_kinds, [spec.name for spec in backend.execution_backends]) for name, backend in list_backends().items()
    } == expected


def test_list_backends_returns_copy():
    backends = list_backends()
    backends.clear()

    assert list_backends()


def test_backend_registration_is_idempotent():
    backend = get_backend("cuda")

    assert register_backend(backend) is backend


def test_backend_rejects_different_redeclaration():
    backend = get_backend("cuda")
    replacement = replace(backend, callbacks={**backend.callbacks, "unit.callback": lambda: None})

    with pytest.raises(ValueError, match="different declaration"):
        register_backend(replacement)


def test_backend_rejects_duplicate_execution_names():
    execution_backend = get_backend("cuda").execution_backends[0]

    with pytest.raises(ValueError, match="must be unique"):
        replace(get_backend("cuda"), execution_backends=(execution_backend, execution_backend))


def test_backend_variants_require_target_predicates():
    backend = replace(get_backend("cuda"), name="unit-cuda", supports_target=None)

    with pytest.raises(ValueError, match="must define supports_target"):
        register_backend(backend)


@pytest.mark.parametrize(
    ("target_kind", "backend_name", "component_name"),
    [
        ("c", "cpu", "c"),
        ("llvm", "cpu", "llvm"),
        ("cuda", "cuda", "cuda"),
        ("hip", "rocm", "hip"),
        ("metal", "metal", "metal"),
        ("webgpu", "webgpu", "webgpu"),
    ],
)
def test_backend_methods_are_the_primary_component_interface(target_kind: str, backend_name: str, component_name: str):
    context = create_backend_context(target_kind, "c", "auto")
    target = context.target
    backend = context.module

    assert backend.name == backend_name
    assert backend.matches(target)
    assert backend.get_device_codegen(target).name == component_name
    assert context.execution_backend.name in backend.allowed_execution_backends(target)
    if any(spec.enable_host_codegen for spec in backend.execution_backends):
        assert backend.get_host_codegen(tvm.target.Target("c")).name == "c"


def test_backend_owns_compile_callbacks():
    assert "tilelang_callback_cuda_validate" in get_backend("cuda").callbacks
    assert "tilelang_callback_cuda_compile" in get_backend("cuda").callbacks
    assert "tilelang_callback_hip_compile" in get_backend("rocm").callbacks
    assert tvm.ffi.get_global_func("tilelang_callback_cuda_compile")
    assert tvm.ffi.get_global_func("tilelang_callback_hip_compile")


def test_cutedsl_backend_reuses_cuda_pipeline():
    cuda_target = tvm.target.Target("cuda")
    cutedsl_target = tvm.target.Target({"kind": "cuda", "keys": ["cuda", "gpu", "cutedsl"]})
    cuda_backend = get_backend("cuda")
    cutedsl_backend = get_backend("cutedsl")

    assert cuda_backend.name == "cuda"
    assert cutedsl_backend.name == "cutedsl"
    assert cutedsl_backend.get_pipeline(cutedsl_target) is cuda_backend.get_pipeline(cuda_target)
    assert cutedsl_backend.get_device_codegen(cutedsl_target).name == "cutedsl"
    assert cutedsl_backend.allowed_execution_backends(cutedsl_target) == ("cutedsl",)


def test_webgpu_only_exposes_tvm_ffi_execution():
    target = tvm.target.Target("webgpu")
    backend = get_backend("webgpu")

    assert backend.allowed_execution_backends(target) == ("tvm_ffi",)


def test_create_backend_context_binds_compile_state():
    context = create_backend_context("cuda", "c", "tvm_ffi")

    assert isinstance(context, BackendContext)
    assert context.module is get_backend("cuda")
    assert context.target.kind.name == "cuda"
    assert context.target_host.kind.name == "c"
    assert context.execution_backend.name == "tvm_ffi"

    with pytest.raises(AttributeError):
        context.target = tvm.target.Target("llvm")
