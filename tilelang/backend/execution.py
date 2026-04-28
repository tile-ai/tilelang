from __future__ import annotations

from typing import Any

from tilelang.backend.common.target import is_cutedsl_target, is_metal_target


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
    if execution_backend == "tvm_ffi":
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        assert artifact.rt_mod is not None, "tvm_ffi backend requires a runtime module."
        return TVMFFIKernelAdapter(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_mod=artifact.host_mod,
            device_mod=artifact.device_mod,
            rt_mod=artifact.rt_mod,
            device_kernel_source=artifact.kernel_source,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    if execution_backend == "cython":
        from tilelang.jit.adapter.cython import CythonKernelAdapter

        return CythonKernelAdapter(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_mod=artifact.host_mod,
            device_mod=artifact.device_mod,
            device_kernel_source=artifact.kernel_source,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    if execution_backend == "nvrtc":
        from tilelang.jit.adapter.nvrtc import NVRTCKernelAdapter

        return NVRTCKernelAdapter(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_mod=artifact.host_mod,
            device_mod=artifact.device_mod,
            device_kernel_source=artifact.kernel_source,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    if execution_backend == "torch":
        from tilelang.jit.adapter.torch import MetalKernelAdapter

        assert is_metal_target(target)
        return MetalKernelAdapter(
            params=params,
            result_idx=result_idx,
            func_or_mod=func_or_mod,
            device_mod=artifact.device_mod,
            kernel_global_source=artifact.kernel_source,
            verbose=verbose,
        )

    if execution_backend == "cutedsl":
        from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

        assert is_cutedsl_target(target)
        return CuTeDSLKernelAdapter(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_mod=artifact.host_mod,
            device_mod=artifact.device_mod,
            device_kernel_source=artifact.kernel_source,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    raise ValueError(f"Invalid execution backend: {execution_backend}")


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
    if execution_backend == "tvm_ffi":
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        return TVMFFIKernelAdapter.from_database(
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

    if execution_backend == "cython":
        from tilelang.jit.adapter.cython import CythonKernelAdapter

        return CythonKernelAdapter.from_database(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            pass_configs=pass_configs,
        )

    if execution_backend == "nvrtc":
        from tilelang.jit.adapter.nvrtc import NVRTCKernelAdapter

        return NVRTCKernelAdapter.from_database(
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

    if execution_backend == "cutedsl":
        from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

        return CuTeDSLKernelAdapter.from_database(
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

    raise ValueError(f"Invalid execution backend: {execution_backend}")
