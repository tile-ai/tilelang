from __future__ import annotations

import os
from typing_extensions import override

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class CuTeDSLKernelCache(KernelCache):
    """Persist native host modules and complete, typed CuTeDSL device artifacts."""

    kernel_lib_path = "kernel.py"
    device_kernel_path = "device_kernel.py"
    host_kernel_path = "host_kernel.py"

    @override
    def _get_memory_cached_kernel(self, key: str) -> JITKernel | None:
        kernel = super()._get_memory_cached_kernel(key)
        if kernel is not None and kernel.adapter.lib_generator.runtime._closed:
            # cleanup releases this adapter, not its reusable disk artifacts.
            # A later compile must load a new runtime instead of returning it.
            del self._memory_cache[key]
            return None
        return kernel

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
        """Publish each already compiled cubin and its symbol metadata together.

        An unexecuted branch may have no cubin yet. The generated native module
        contains its full compilation input, so those variants remain lazy.
        """
        kernel.adapter.lib_generator.runtime.export_artifacts(cache_path)

    @override
    def _set_adapter_cache_path(self, kernel: JITKernel, cache_path: str):
        """Persist newly reached device variants after subsequent executions."""
        if hasattr(kernel, "adapter"):
            kernel.adapter._cache_path = cache_path
