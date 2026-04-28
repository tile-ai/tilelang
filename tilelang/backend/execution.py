from __future__ import annotations

from typing import Any

from tvm.target import Target

from tilelang.backend.base import ExecutionBackendSpec, LibraryCompileSpec
from tilelang.backend.registry import get_device_backend, resolve_execution_backend


def get_execution_spec(target: Target, execution_backend: str | None) -> ExecutionBackendSpec:
    resolved = resolve_execution_backend(execution_backend, target)
    return get_device_backend(target).execution_spec(resolved)


def _first_execution_spec_with(target: Target, attr_name: str) -> ExecutionBackendSpec:
    backend = get_device_backend(target)
    for spec in backend.execution_specs:
        if getattr(spec, attr_name) is not None:
            return spec
    raise ValueError(f"Backend '{backend.name}' does not provide execution metadata '{attr_name}'")


def create_execution_adapter(
    execution_backend: str,
    *,
    artifact,
    params,
    result_idx,
    target,
    func_or_mod,
    verbose: bool,
    pass_configs: dict[str, Any] | None,
    compile_flags: list[str] | None,
):
    spec = get_execution_spec(target, execution_backend)
    return spec.adapter_factory(
        artifact=artifact,
        params=params,
        result_idx=result_idx,
        target=target,
        func_or_mod=func_or_mod,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def create_execution_adapter_from_database(
    execution_backend: str,
    *,
    params,
    result_idx,
    target,
    func_or_mod,
    host_kernel_source: str,
    device_kernel_source: str,
    kernel_lib_path: str,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
):
    spec = get_execution_spec(target, execution_backend)
    if spec.database_adapter_factory is None:
        raise ValueError(f"Execution backend '{execution_backend}' does not support database cache loading")
    return spec.database_adapter_factory(
        params=params,
        result_idx=result_idx,
        target=target,
        func_or_mod=func_or_mod,
        host_kernel_source=host_kernel_source,
        device_kernel_source=device_kernel_source,
        kernel_lib_path=kernel_lib_path,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def create_c_source_wrapper(device_target: Target, **kwargs):
    spec = _first_execution_spec_with(device_target, "c_source_wrapper_factory")
    return spec.c_source_wrapper_factory(**kwargs)


def create_python_source_wrapper(device_target: Target, **kwargs):
    spec = _first_execution_spec_with(device_target, "python_source_wrapper_factory")
    return spec.python_source_wrapper_factory(**kwargs)


def get_library_compile_spec(target: Target) -> LibraryCompileSpec:
    spec = _first_execution_spec_with(target, "library_compile_spec")
    return spec.library_compile_spec


def get_kernel_cache(target: Target, execution_backend: str):
    spec = get_execution_spec(target, execution_backend)
    return spec.cache_factory()
