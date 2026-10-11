import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TVMFFIKernelCache(KernelCache):
    kernel_lib_path = "executable.so"

    def _get_kernel_lib_path(self, cache_path: str) -> str:
        binary = os.path.join(cache_path, "executable.tlm")
        return binary if os.path.isfile(binary) else super()._get_kernel_lib_path(cache_path)

    def _save_kernel_to_disk(self, key, kernel, func=None, verbose=False):
        super()._save_kernel_to_disk(key, kernel, func, verbose)
        library = self._get_kernel_lib_path(self._get_cache_path(key))
        if os.path.isfile(library):
            if library.endswith(".tlm"):
                # MIR is already executable; reloading would JIT it a second time.
                kernel.adapter.libpath = library
                return
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
        name = "executable.tlm" if kernel.target_host.kind.name == "mir" else self.kernel_lib_path
        kernel_lib_path = os.path.join(cache_path, name)
        if verbose:
            self.logger.debug(f"Saving kernel executable to file: {kernel_lib_path}")
        self.export_library(kernel, kernel_lib_path)

    @staticmethod
    def export_library(kernel: JITKernel, path: str):
        """Export fresh modules or copy loaded libraries through the same writer."""
        library = getattr(kernel.adapter, "libpath", None)
        if library and os.path.splitext(library)[1] == os.path.splitext(path)[1]:
            if not (os.path.exists(path) and os.path.samefile(library, path)):
                KernelCache._safe_write_file(path, "wb", lambda file: file.write(KernelCache._load_binary(library)))
        elif path.endswith(".tlm"):
            from tvm import ffi, runtime

            executable = kernel.adapter.get_exportable_executable()
            module = executable.mod if isinstance(executable, runtime.Executable) else executable
            data = ffi.get_global_func("tilelang.ModuleSerialize")(module)
            KernelCache._safe_write_file(path, "wb", lambda file: file.write(data))
        else:
            KernelCache._safe_write_executable(
                kernel.adapter.get_exportable_executable(), path, export_kwargs=TVMFFIKernelCache._get_export_kwargs(kernel)
            )
