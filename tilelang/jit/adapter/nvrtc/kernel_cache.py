import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class NVRTCKernelCache(KernelCache):
    kernel_lib_path = "kernel.cubin"
    kernel_py_path = "kernel.py"

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        src_lib_path = kernel.adapter.libpath
        kernel_py_path = os.path.join(cache_path, self.kernel_py_path)
        src_lib_path = src_lib_path.replace(".cubin", ".py")
        if verbose:
            self.logger.debug(f"Saving kernel nvrtc python code to file: {kernel_py_path}")
        KernelCache._safe_write_file(kernel_py_path, "wb", lambda file: file.write(KernelCache._load_binary(src_lib_path)))
        super()._save_so_cubin_to_disk(kernel, cache_path, verbose)
