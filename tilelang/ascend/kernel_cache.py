"""Ascend-specific full-kernel cache layouts."""

from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache


class AscendTVMFFIKernelCache(TVMFFIKernelCache):
    """Use source suffixes that describe the exported Ascend artifacts."""

    _instance = None
    device_kernel_path = "device_kernel.asc"
    host_kernel_path = "host_kernel.c"


class AscendCythonKernelCache(CythonKernelCache):
    """Cache the Bisheng device and combined launcher sources as AscendC."""

    _instance = None
    device_kernel_path = "device_kernel.asc"
    host_kernel_path = "host_kernel.asc"
