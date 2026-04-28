from __future__ import annotations

from typing import Any

from tvm.target import Target

from tilelang.backend.base import (
    CacheArtifactSpec,
    ExecutionBackendSpec,
    LibraryCompileSpec,
    SourceWrapperFactory,
)


CPU_EXECUTION_BACKENDS = ("cython", "tvm_ffi")
CPU_DEFAULT_EXECUTION_BACKEND = "cython"


def _create_tvm_ffi_adapter(**kwargs):
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    artifact = kwargs.pop("artifact")
    assert artifact.rt_mod is not None, "tvm_ffi backend requires a runtime module."
    return TVMFFIKernelAdapter(
        params=kwargs["params"],
        result_idx=kwargs["result_idx"],
        target=kwargs["target"],
        func_or_mod=kwargs["func_or_mod"],
        host_mod=artifact.host_mod,
        device_mod=artifact.device_mod,
        rt_mod=artifact.rt_mod,
        device_kernel_source=artifact.kernel_source,
        verbose=kwargs["verbose"],
        pass_configs=kwargs["pass_configs"],
        compile_flags=kwargs["compile_flags"],
    )


def _create_tvm_ffi_adapter_from_database(**kwargs):
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    return TVMFFIKernelAdapter.from_database(**kwargs)


def _create_tvm_ffi_cache():
    from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache

    return TVMFFIKernelCache()


def _create_cython_adapter(**kwargs):
    from tilelang.jit.adapter.cython import CythonKernelAdapter

    artifact = kwargs.pop("artifact")
    return CythonKernelAdapter(
        params=kwargs["params"],
        result_idx=kwargs["result_idx"],
        target=kwargs["target"],
        func_or_mod=kwargs["func_or_mod"],
        host_mod=artifact.host_mod,
        device_mod=artifact.device_mod,
        device_kernel_source=artifact.kernel_source,
        verbose=kwargs["verbose"],
        pass_configs=kwargs["pass_configs"],
        compile_flags=kwargs["compile_flags"],
    )


def _create_cython_adapter_from_database(**kwargs):
    from tilelang.jit.adapter.cython import CythonKernelAdapter

    return CythonKernelAdapter.from_database(**kwargs)


def _create_cython_cache():
    from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache

    return CythonKernelCache()


def _cpu_source_wrapper(**kwargs):
    from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper

    return TLCPUSourceWrapper(**kwargs)


def _cpu_library_command(target: Target, source_path: str, library_path: str, pass_configs: dict[str, Any]) -> list[str]:
    from tilelang.contrib.cc import get_cplus_compiler
    from tilelang.env import TILELANG_TEMPLATE_PATH

    return [
        get_cplus_compiler(),
        "-std=c++17",
        "-fPIC",
        "-shared",
        source_path,
        "-I" + TILELANG_TEMPLATE_PATH,
        "-o",
        library_path,
    ]


CPU_LIBRARY_COMPILE_SPEC = LibraryCompileSpec(
    source_suffix=".cpp",
    library_suffix=".so",
    command_factory=_cpu_library_command,
)


def make_tvm_ffi_execution_spec() -> ExecutionBackendSpec:
    return ExecutionBackendSpec(
        name="tvm_ffi",
        adapter_factory=_create_tvm_ffi_adapter,
        database_adapter_factory=_create_tvm_ffi_adapter_from_database,
        cache_factory=_create_tvm_ffi_cache,
        cache_artifact=CacheArtifactSpec(kernel_lib_path="executable.so"),
        requires_host_codegen=True,
        requires_device_compile=True,
    )


def make_cython_execution_spec(
    *,
    c_source_wrapper_factory: SourceWrapperFactory | None,
    library_compile_spec: LibraryCompileSpec | None,
) -> ExecutionBackendSpec:
    return ExecutionBackendSpec(
        name="cython",
        adapter_factory=_create_cython_adapter,
        database_adapter_factory=_create_cython_adapter_from_database,
        cache_factory=_create_cython_cache,
        c_source_wrapper_factory=c_source_wrapper_factory,
        library_compile_spec=library_compile_spec,
        requires_cxx_compiler=True,
    )


CPU_EXECUTION_SPECS = (
    make_cython_execution_spec(
        c_source_wrapper_factory=_cpu_source_wrapper,
        library_compile_spec=CPU_LIBRARY_COMPILE_SPEC,
    ),
    make_tvm_ffi_execution_spec(),
)
