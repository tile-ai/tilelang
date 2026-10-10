import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TVMFFIKernelCache(KernelCache):
    kernel_lib_path = "executable.so"

    def _save_kernel_to_disk(self, key, kernel, func=None, verbose=False):
        super()._save_kernel_to_disk(key, kernel, func, verbose)
        library = os.path.join(self._get_cache_path(key), self.kernel_lib_path)
        if os.path.isfile(library):
            from tvm import runtime

            # Load only after publication; do not lock the staging DLL on Windows.
            try:
                kernel.adapter.executable = runtime.load_module(library)
            except Exception:
                self.logger.warning("Could not reuse the published Host IR library", exc_info=True)
                return
            kernel.adapter.libpath = library

    @staticmethod
    def _get_export_kwargs(kernel: JITKernel) -> dict:
        artifact = getattr(kernel, "artifact", None)
        target_host = getattr(artifact, "target_host", None)
        target_host_kind = getattr(getattr(target_host, "kind", None), "name", None)
        if target_host_kind == "c":
            return KernelCache._get_source_compile_args()
        return KernelCache._get_export_link_args()

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.adapter.get_host_source()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        if verbose:
            self.logger.debug(f"Saving kernel executable to file: {kernel_lib_path}")
        self.export_library(kernel, kernel_lib_path)

    @staticmethod
    def export_library(kernel: JITKernel, path: str):
        """Export fresh modules or copy loaded libraries through the same writer."""
        library = getattr(kernel.adapter, "libpath", None)
        if library:
            if not (os.path.exists(path) and os.path.samefile(library, path)):
                KernelCache._safe_write_file(path, "wb", lambda file: file.write(KernelCache._load_binary(library)))
        else:
            KernelCache._safe_write_executable(
                kernel.adapter.get_exportable_executable(), path, export_kwargs=TVMFFIKernelCache._get_export_kwargs(kernel)
            )
