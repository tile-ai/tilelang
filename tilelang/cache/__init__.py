"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from tilelang import env
from .auto_kernel_cache import AutoKernelCache
from .ctypes_kernel_cache import CTypesKernelCache
from .cutedsl_kernel_cache import CuTeDSLKernelCache
from .cython_kernel_cache import CythonKernelCache
from .nvrtc_kernel_cache import NVRTCKernelCache
from .torch_kernel_cache import TorchKernelCache
from .tvm_ffi_kernel_cache import TVMFFIKernelCache

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

# Create a pool of singleton instance of KernelCaches
_dispatch_pool: dict[str, KernelCache] = {
    "auto": AutoKernelCache(),
    "tvm_ffi": TVMFFIKernelCache(),
    "ctypes": CTypesKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "torch": TorchKernelCache(),
}


def cached(
    func: PrimFunc = None,
    out_idx: list[int] = None,
    *args,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    return _dispatch_pool[execution_backend].cached(
        func,
        out_idx,
        *args,
        target=target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def clear_cache():
    """
    Disabled helper that previously removed the entire kernel cache.

    Raises:
        RuntimeError: Always raised to warn users to clear the cache manually.
    """
    cache_dir = env.TILELANG_CACHE_DIR
    raise RuntimeError(
        "tilelang.clear_cache() is disabled because deleting the cache directory "
        "is dangerous. If you accept the risk, remove it manually with "
        f"`rm -rf '{cache_dir}'`."
    )


if env.TILELANG_CLEAR_CACHE.lower() in ("1", "true", "yes", "on"):
    clear_cache()
