# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - Init file"""

from typing import List, Union, Literal, Optional
from pathlib import Path
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from .kernel_cache import KernelCache
from tilelang.env import TILELANG_CLEAR_CACHE

# Create singleton instance of KernelCache
_kernel_cache_instance = KernelCache()


def cached(
    func: PrimFunc = None,
    out_idx: List[int] = None,
    *args,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    execution_backend: Optional[Literal["dlpack", "ctypes", "cython"]] = "cython",
    verbose: Optional[bool] = False,
    pass_configs: Optional[dict] = None,
) -> JITKernel:
    """
    Caches and reuses compiled kerne(ls (using KernelCache class).
    """
    return _kernel_cache_instance.cached(
        func,
        out_idx,
        *args,
        target=target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
    )


def get_cache_dir() -> Path:
    """
    Gets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.get_cache_dir()
        PosixPath('/Users/username/.tilelang/cache')
    """
    return _kernel_cache_instance.get_cache_dir()


def set_cache_dir(cache_dir: str):
    """
    Sets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.set_cache_dir("/path/to/cache")
    """
    _kernel_cache_instance.set_cache_dir(cache_dir)


def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()


if TILELANG_CLEAR_CACHE.lower() in ("1", "true", "yes", "on"):
    clear_cache()
