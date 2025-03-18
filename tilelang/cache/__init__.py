# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - Init file"""

from typing import Callable, List, Union
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from .kernel_cache import KernelCache
from typing import Callable, List, Literal, Union, Optional

# Create singleton instance of KernelCache
_kernel_cache_instance = KernelCache()


def cached(
    func: PrimFunc=None,
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
    print(type(func))
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


def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()
