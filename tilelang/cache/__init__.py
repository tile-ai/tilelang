# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - Init file"""

from typing import Callable, List, Union
from tvm.target import Target
from tilelang.jit import JITKernel
from .kernel_cache import KernelCache

# Create singleton instance of KernelCache
_kernel_cache_instance = KernelCache()


def cached(
    func: Callable,
    out_idx: List[int] = None,
    *args,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    return _kernel_cache_instance.cached_kernel(
        func, out_idx, *args, target=target, target_host=target_host)


def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()
