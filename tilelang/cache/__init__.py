# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils"""

from tilelang import compile
from tilelang.jit import JITKernel
from typing import Callable, List, Union
from tvm.target import Target
from tvm.tir import PrimFunc

# Dictionary to store cached kernels
_cached = {}


def cached(
    func: Callable,
    out_idx: List[int] = None,
    *args,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
) -> JITKernel:
    """
    Cache and reuse compiled kernels to avoid redundant compilation.
    
    Args:
        func: Function to be compiled or a PrimFunc that's already prepared
        out_idx: Indices specifying which outputs to return
        target: Compilation target platform
        target_host: Host target for compilation
        *args: Arguments passed to func when calling it
        
    Returns:
        JITKernel: The compiled kernel, either freshly compiled or from cache
    """
    global _cached
    # Create a unique key based on the function, output indices and arguments
    key = (func, tuple(out_idx), *args)

    # Return cached kernel if available
    if key not in _cached:
        # Handle both PrimFunc objects and callable functions
        program = func if isinstance(func, PrimFunc) else func(*args)

        # Compile the program to a kernel
        kernel = compile(program, out_idx=out_idx, target=target, target_host=target_host)
        # Store in cache for future use
        _cached[key] = kernel

    return _cached[key]


def clear_cache():
    """
    Clear the entire kernel cache.
    
    This function resets the internal cache dictionary that stores compiled kernels.
    Use this when you want to free memory or ensure fresh compilation
    of kernels in a new context.
    """
    global _cached
    _cached = {}
