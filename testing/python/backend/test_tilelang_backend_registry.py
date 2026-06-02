from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, list_backends, resolve_backend
from tilelang.jit.execution_backend import allowed_backends_for_target, resolve_execution_backend


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
