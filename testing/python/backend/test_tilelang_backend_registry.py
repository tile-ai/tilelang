from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, list_backends, resolve_backend
import tilelang.backend.registry as backend_registry
from tilelang.engine.lower import is_cpu_device_backend
from tilelang.jit.execution_backend import allowed_backends_for_target, resolve_execution_backend
from tilelang.jit.kernel import JITKernel


def test_backend_descriptor_accepts_callable_pipeline():
    target = tvm.target.Target("c")
    mod = tvm.IRModule({})
    backend = Backend("unit-test", ("c",), pipeline=lambda input_mod, input_target: input_mod)

    assert backend.matches(target)
    assert backend.lower(mod, target) is mod


def test_builtin_backend_resolution():
    expected = {
        "c": "cpu",
        "llvm": "cpu",
        "cuda": "cuda",
        "hip": "rocm",
        "metal": "metal",
        "webgpu": "webgpu",
    }

    for target_kind, backend_name in expected.items():
        assert resolve_backend(tvm.target.Target(target_kind)).name == backend_name


def test_missing_codegen_hook_reports_backend_name():
    backend = Backend("unit-test", ("c",))

    with pytest.raises(ValueError, match="unit-test"):
        backend.codegen(tvm.IRModule({}), tvm.target.Target("c"), compile=False)


def test_list_backends_returns_copy():
    registered = list_backends()
    registered.clear()

    assert list_backends()


def test_backend_descriptor_freezes_nested_mappings():
    backend = Backend(
        "unit-test",
        ("c",),
        features={"feature": lambda target: True},
        execution_backends={"fast": ExecutionBackendSpec("fast")},
    )

    with pytest.raises(TypeError):
        backend.features["other"] = lambda target: False
    with pytest.raises(TypeError):
        backend.execution_backends["slow"] = ExecutionBackendSpec("slow")


def test_backend_override_prunes_old_target_bucket():
    backend_name = "unit-prune"
    old_kind = "unit-old"
    new_kind = "unit-new"

    try:
        backend_registry.register_backend(Backend(backend_name, (old_kind,)), override=True)
        assert backend_name in backend_registry._TARGET_INDEX[old_kind]

        backend_registry.register_backend(Backend(backend_name, (new_kind,)), override=True)

        assert old_kind not in backend_registry._TARGET_INDEX
        assert backend_name in backend_registry._TARGET_INDEX[new_kind]
    finally:
        backend_registry._BACKENDS.pop(backend_name, None)
        backend_registry._CALLBACKS_REGISTERED.discard(backend_name)
        for kind in (old_kind, new_kind):
            names = backend_registry._TARGET_INDEX.get(kind, [])
            if backend_name in names:
                names.remove(backend_name)
            if not names:
                backend_registry._TARGET_INDEX.pop(kind, None)


def test_backend_resolves_execution_backend_policy():
    target = tvm.target.Target("c")
    backend = Backend(
        "unit-test",
        ("c",),
        execution_backends={
            "slow": ExecutionBackendSpec("slow", adapter="cython"),
            "fast": ExecutionBackendSpec(
                "fast",
                adapter="tvm_ffi",
                enable_host_codegen=True,
                enable_device_compile=True,
            ),
        },
        default_execution_backend="fast",
    )

    spec = backend.resolve_execution_backend("auto", target)

    assert spec.name == "fast"
    assert spec.adapter == "tvm_ffi"
    assert spec.enable_host_codegen
    assert spec.enable_device_compile


def test_execution_backend_resolves_through_backend_descriptor():
    expected = {
        "c": "cython",
        "llvm": "cython",
        "cuda": "tvm_ffi",
        "hip": "tvm_ffi",
        "metal": "tvm_ffi",
        "webgpu": "tvm_ffi",
    }

    for target_kind, execution_backend in expected.items():
        assert resolve_execution_backend("auto", tvm.target.Target(target_kind)) == execution_backend


def test_invalid_execution_backend_reports_resolved_backend():
    with pytest.raises(ValueError, match="Backend 'rocm'"):
        resolve_execution_backend("nvrtc", tvm.target.Target("hip"))


def test_cuda_only_execution_backend_policy_is_backend_owned():
    target = tvm.target.Target("cuda")
    backend = resolve_backend(target)

    assert "nvrtc" in allowed_backends_for_target(target)
    assert "nvrtc" in backend.allowed_execution_backends(target)
    assert "nvrtc" not in resolve_backend(tvm.target.Target("hip")).allowed_execution_backends(tvm.target.Target("hip"))


def test_llvm_is_cpu_device_backend():
    assert is_cpu_device_backend(tvm.target.Target("c"))
    assert is_cpu_device_backend(tvm.target.Target("llvm"))


def test_jit_source_helpers_use_resolved_adapter_key():
    class FakeAdapter:
        def get_kernel_source(self, kernel_only=True):
            return f"kernel:{kernel_only}"

        def get_host_source(self):
            return "host"

    kernel = JITKernel.__new__(JITKernel)
    kernel.execution_backend = "fast"
    kernel.execution_backend_spec = ExecutionBackendSpec("fast", adapter="tvm_ffi")
    kernel.adapter = FakeAdapter()
    kernel.artifact = None

    assert kernel.get_kernel_source(kernel_only=True) == "kernel:True"
    assert kernel.get_host_source() == "host"
