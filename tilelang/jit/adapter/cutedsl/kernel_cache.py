from __future__ import annotations

import os
from typing import Callable, Literal
from typing_extensions import override

from tvm.target import Target
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam
from tilelang.jit import JITKernel


class CuTeDSLKernelCache(KernelCache):
    # CuTeDSL C++ launcher specific
    kernel_lib_path = "kernel.py"
    device_kernel_path = "kernel.py"
    host_kernel_path = "kernel.py"
    launcher_lib_path = "launcher_lib.so"
    launcher_cpp_path = "launcher.cpp"

    @override
    def _save_kernel_source_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        return

    @override
    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        # Save C++ launcher library if it exists
        lib_gen = getattr(kernel.adapter, "lib_generator", None)
        if lib_gen and hasattr(lib_gen, "launcher_libpath") and lib_gen.launcher_libpath:
            launcher_lib_path = os.path.join(cache_path, self.launcher_lib_path)
            src_launcher_path = lib_gen.launcher_libpath
            if verbose:
                self.logger.debug(f"Saving C++ launcher library to cache: {src_launcher_path}")
            KernelCache._safe_write_file(launcher_lib_path, "wb", lambda file: file.write(KernelCache._load_binary(src_launcher_path)))

        # Optionally save launcher C++ source for debugging
        if hasattr(kernel.adapter, "launcher_cpp_code") and kernel.adapter.launcher_cpp_code:
            launcher_cpp_path = os.path.join(cache_path, self.launcher_cpp_path)
            if verbose:
                self.logger.debug(f"Saving C++ launcher source to: {launcher_cpp_path}")
            KernelCache._safe_write_file(launcher_cpp_path, "w", lambda file: file.write(kernel.adapter.launcher_cpp_code))

    @override
    def _get_required_files(self, cache_path: str) -> list[str]:
        return super()._get_required_files(cache_path) + [os.path.join(cache_path, self.launcher_lib_path)]


    @override
    def _set_adapter_cache_path(self, kernel: JITKernel, cache_path: str):
        if hasattr(kernel, "adapter"):
            kernel.adapter._cache_path = cache_path

    @override
    def _build_kernel(
        self,
        func: Callable | None,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        kernel_params: list[KernelParam] | None,
        target: str | Target,
        target_host: str | Target | None,
        out_idx: list[int] | None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"],
        pass_configs: dict | None,
        compile_flags: list[str] | str | None,
    ) -> JITKernel | None:
        if kernel_params:
            return JITKernel.from_database(
                func=func,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=target,
                target_host=target_host,
                out_idx=out_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            return None
