"""Disk cache integration for CUDA Tile IR backend artifacts."""

from __future__ import annotations

import os
from typing_extensions import override

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel
from tilelang.tileir.artifact import TILEIR_CACHE_FILENAME


class TileIRKernelCache(KernelCache):
    """Disk cache layout for CUDA Tile IR backend artifacts."""

    device_kernel_path = "kernel.tileir"
    host_kernel_path = "tileir_launcher.py"
    kernel_lib_path = TILEIR_CACHE_FILENAME

    @override
    def _save_kernel_source_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Save the structured TileIR source emitted by the lowering path."""
        device_kernel_path = os.path.join(cache_path, self.device_kernel_path)
        if verbose:
            self.logger.debug(f"Saving TileIR kernel source to file: {device_kernel_path}")
        KernelCache._safe_write_file(device_kernel_path, "w", lambda file: file.write(kernel.get_kernel_source()))

    @override
    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Save the native-dispatcher launcher metadata for source inspection."""
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving TileIR launcher source to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.get_host_source()))

    @override
    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        """Persist the versioned, data-only TileIR runtime artifact."""
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        if verbose:
            self.logger.debug(f"Saving TileIR runtime artifact to file: {kernel_lib_path}")
        payload = kernel.adapter._serialize_tileir_artifact(kernel.adapter.tileir_artifact)
        KernelCache._safe_write_file(kernel_lib_path, "wb", lambda file: file.write(payload))
