from __future__ import annotations

import os
from typing_extensions import override

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class CuTeDSLKernelCache(KernelCache):
    """Persist CuTeDSL kernels as Python modules plus launcher artifacts."""

    # CuTeDSL C++ launcher specific
    kernel_lib_path = "kernel.py"
    device_kernel_path = "device_kernel.py"
    host_kernel_path = "host_kernel.py"
    launcher_lib_path = "launcher_lib.so"
    launcher_cpp_path = "launcher.cpp"

    @override
    def _save_kernel_source_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Save the original CuTeDSL device source for source inspection."""
        device_kernel_path = os.path.join(cache_path, self.device_kernel_path)
        if verbose:
            self.logger.debug(f"Saving CuTeDSL device source to file: {device_kernel_path}")
        device_source = kernel.adapter.get_kernel_source(kernel_only=True)
        if device_source is not None:
            KernelCache._safe_write_file(device_kernel_path, "w", lambda file: file.write(device_source))

    @override
    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Save both the generated host wrapper and importable kernel module."""
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving CuTeDSL host source to file: {host_kernel_path}")
        host_source = kernel.adapter.get_host_source()
        if host_source is not None:
            KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(host_source))

        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        if verbose:
            self.logger.debug(f"Saving CuTeDSL generated module to file: {kernel_lib_path}")
        generated_module_source = kernel.adapter.get_generated_module_source()
        if generated_module_source is None:
            generated_module_source = KernelCache._load_binary(kernel.adapter.libpath).decode()
        KernelCache._safe_write_file(kernel_lib_path, "w", lambda file: file.write(generated_module_source))

    @override
    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Save launcher shared objects and optional debugging artifacts."""
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
        """Return the CuTeDSL cache files required for a runnable reload."""
        return super()._get_required_files(cache_path) + [os.path.join(cache_path, self.launcher_lib_path)]

    @override
    def _set_adapter_cache_path(self, kernel: JITKernel, cache_path: str):
        """Record the cache path so first execution can persist generated cubin."""
        if hasattr(kernel, "adapter"):
            kernel.adapter._cache_path = cache_path
