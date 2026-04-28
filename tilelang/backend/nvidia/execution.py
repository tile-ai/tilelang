from __future__ import annotations

from typing import Any

from tvm.target import Target

from tilelang.backend.base import CacheArtifactSpec, ExecutionBackendSpec, LibraryCompileSpec
from tilelang.backend.common.execution import make_cython_execution_spec, make_tvm_ffi_execution_spec
from tilelang.transform import PassConfigKey


CUDA_DEFAULT_EXECUTION_BACKEND = "tvm_ffi"
CUTEDSL_DEFAULT_EXECUTION_BACKEND = "cutedsl"


def _cuda_source_wrapper(**kwargs):
    from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper

    return TLCUDASourceWrapper(**kwargs)


def _cuda_nvrtc_source_wrapper(**kwargs):
    from tilelang.jit.adapter.nvrtc import TLNVRTCSourceWrapper

    return TLNVRTCSourceWrapper(**kwargs)


def _cutedsl_source_wrapper(**kwargs):
    from tilelang.jit.adapter.cutedsl import TLCuTeDSLSourceWrapper

    return TLCuTeDSLSourceWrapper(**kwargs)


def _cuda_library_command(target: Target, source_path: str, library_path: str, pass_configs: dict[str, Any]) -> list[str]:
    from tilelang.contrib.nvcc import get_nvcc_compiler, get_target_arch, get_target_compute_version
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    target_arch = get_target_arch(get_target_compute_version(target))
    command = [
        get_nvcc_compiler(),
        "-std=c++17",
        "-w",
        "-Xcudafe",
        "--diag_suppress=177",
        "--compiler-options",
        "-fPIC",
        "-lineinfo",
        "--shared",
        source_path,
        "-lcuda",
        "-gencode",
        f"arch=compute_{target_arch},code=sm_{target_arch}",
    ]

    if pass_configs.get(PassConfigKey.TL_ENABLE_FAST_MATH, False):
        command += ["--use_fast_math"]

    ptxas_usage_level = pass_configs.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    if ptxas_usage_level is not None:
        command += [f"--ptxas-options=--register-usage-level={int(ptxas_usage_level)}"]

    if pass_configs.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False):
        command += ["--ptxas-options=--verbose"]

    command += [
        "-I" + CUTLASS_INCLUDE_DIR,
        "-I" + TILELANG_TEMPLATE_PATH,
        "-o",
        library_path,
    ]
    return command


CUDA_LIBRARY_COMPILE_SPEC = LibraryCompileSpec(
    source_suffix=".cu",
    library_suffix=".so",
    command_factory=_cuda_library_command,
)


def _create_nvrtc_adapter(**kwargs):
    from tilelang.jit.adapter.nvrtc import NVRTCKernelAdapter

    artifact = kwargs.pop("artifact")
    return NVRTCKernelAdapter(
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


def _create_nvrtc_adapter_from_database(**kwargs):
    from tilelang.jit.adapter.nvrtc import NVRTCKernelAdapter

    return NVRTCKernelAdapter.from_database(**kwargs)


def _create_nvrtc_cache():
    from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache

    return NVRTCKernelCache()


def _create_cutedsl_adapter(**kwargs):
    from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

    artifact = kwargs.pop("artifact")
    return CuTeDSLKernelAdapter(
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


def _create_cutedsl_adapter_from_database(**kwargs):
    from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

    return CuTeDSLKernelAdapter.from_database(**kwargs)


def _create_cutedsl_cache():
    from tilelang.jit.adapter.cutedsl.kernel_cache import CuTeDSLKernelCache

    return CuTeDSLKernelCache()


CUDA_EXECUTION_SPECS = (
    make_tvm_ffi_execution_spec(),
    ExecutionBackendSpec(
        name="nvrtc",
        adapter_factory=_create_nvrtc_adapter,
        database_adapter_factory=_create_nvrtc_adapter_from_database,
        cache_factory=_create_nvrtc_cache,
        cache_artifact=CacheArtifactSpec(kernel_lib_path="kernel.cubin", extra_required_paths=("kernel.py",)),
        python_source_wrapper_factory=_cuda_nvrtc_source_wrapper,
    ),
    make_cython_execution_spec(
        c_source_wrapper_factory=_cuda_source_wrapper,
        library_compile_spec=CUDA_LIBRARY_COMPILE_SPEC,
    ),
)
CUDA_EXECUTION_BACKENDS = tuple(spec.name for spec in CUDA_EXECUTION_SPECS)

CUTEDSL_EXECUTION_SPECS = (
    ExecutionBackendSpec(
        name="cutedsl",
        adapter_factory=_create_cutedsl_adapter,
        database_adapter_factory=_create_cutedsl_adapter_from_database,
        cache_factory=_create_cutedsl_cache,
        cache_artifact=CacheArtifactSpec(
            kernel_lib_path="kernel.py",
            device_kernel_path="kernel.py",
            host_kernel_path="kernel.py",
            extra_required_paths=("launcher_lib.so",),
        ),
        python_source_wrapper_factory=_cutedsl_source_wrapper,
    ),
)
CUTEDSL_EXECUTION_BACKENDS = tuple(spec.name for spec in CUTEDSL_EXECUTION_SPECS)


def unavailable_cuda_execution_backends() -> tuple[str, ...]:
    try:
        from tilelang.jit.adapter.nvrtc import is_nvrtc_available
    except Exception:
        return ()

    return () if is_nvrtc_available else ("nvrtc",)
