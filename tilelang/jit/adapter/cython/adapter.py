# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from ..base import BaseKernelAdapter
import ctypes
from typing import List, Optional, Union, Callable, Dict, Tuple
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.relay import TensorType
from tvm import tir
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.tensor import map_torch_type
from tilelang.contrib.cc import get_cplus_compiler
import torch
import sys
import sysconfig
import hashlib
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_cython_compiler() -> Optional[str]:
    """Return the path to the Cython compiler.

    Returns
    -------
    out: Optional[str]
        The path to the Cython compiler, or None if none was found.
    """

    cython_names = ["cython", "cython3"]
    dirs_in_path = os.get_exec_path()
    for cython_name in cython_names:
        for d in dirs_in_path:
            cython_path = os.path.join(d, cython_name)
            if os.path.isfile(cython_path) and os.access(cython_path, os.X_OK):
                return cython_path
    return None


# Add cache management functions at module level
def get_cache_dir() -> Path:
    """Get the cache directory for the current Python version."""
    py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    # current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = Path(current_dir) / ".cycache" / py_version
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_lib(source_code: str) -> Tuple[Optional[ctypes.CDLL], Path]:
    """Try to load cached library or return None if not found."""
    code_hash = hashlib.sha256(source_code.encode()).hexdigest()
    cache_path = get_cache_dir() / f"{code_hash}.so"
    if cache_path.exists():
        try:
            return ctypes.CDLL(str(cache_path)), cache_path
        except Exception as e:
            logger.error(f"Failed to load cached library: {e}")
            return None, cache_path
    return None, cache_path


# read the cython_wrapper.pyx file
current_dir = os.path.dirname(os.path.abspath(__file__))
cython_wrapper_path = os.path.join(current_dir, "cython_wrapper.pyx")

with open(cython_wrapper_path, "r") as f:
    cython_wrapper_code = f.read()
    cache_dir = get_cache_dir()
    source_path = cache_dir / "cython_wrapper.cpp"
    library_path = cache_dir / "cython_wrapper.so"
    md5_path = cache_dir / "md5.txt"
    code_hash = hashlib.sha256(cython_wrapper_code.encode()).hexdigest()

    # Check if cached version exists and is valid
    need_compile = True
    if md5_path.exists() and library_path.exists():
        with open(md5_path, "r") as f:
            cached_hash = f.read().strip()
            if cached_hash == code_hash:
                logger.debug("Cython jit adapter is up to date, no need to compile...")
                need_compile = False
            else:
                logger.info("Cython jit adapter is out of date, need to recompile...")
    else:
        logger.info("No cached version found for cython jit adapter, need to compile...")

    if need_compile:
        logger.info("Compiling cython jit adapter...")
        with open(md5_path, "w") as f:
            f.write(code_hash)
        # compile the cython_wrapper.pyx file into .cpp
        cython = get_cython_compiler()
        if cython is None:
            raise Exception("Cython is not installed, please install it first.")
        os.system(f"{cython} {cython_wrapper_path} --cplus -o {source_path}")
        # compile the .cpp file into .so
        python_include_path = sysconfig.get_path("include")
        cc = get_cplus_compiler()
        command = f"{cc} -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I{python_include_path} {source_path} -o {library_path}"
        try:
            os.system(command)
        except Exception as e:
            raise Exception(f"Failed to compile cython jit adapter: {e}") from e

    # add the .so file to the sys.path
    cache_dir_str = str(cache_dir)
    if cache_dir_str not in sys.path:
        sys.path.append(cache_dir_str)

from cython_wrapper import CythonKernelWrapper


class CythonKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using ctypes.
    
    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target: str = "cuda"
    ir_module: Optional[tvm.IRModule] = None
    lib: Optional[ctypes.CDLL] = None  # Compiled library handle
    wrapped_source: Optional[str] = None  # Generated C++ wrapper code
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: Optional[Dict[tir.Var, Tuple[int, torch.dtype]]] = None
    # Maps buffer variables to their corresponding static shapes
    # {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape = (16, 16)
    # }
    static_shape_map: Optional[Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]]] = None

    def __init__(self,
                 rt_mod,
                 params: List[TensorType],
                 result_idx: List[int],
                 target,
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 verbose: bool = False):
        """Initialize the adapter with the given TIR function or module.
        
        Args:
            rt_mod: Runtime module
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.mod = rt_mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.static_shape_map = self._process_static_shape()

        self.target = Target.canon_target(determine_target(target))
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()

        try:
            self.lib.init()
        except Exception as e:
            raise Exception(
                f"Failed to initialize the compiled library for {self.target}: {e}") from e

        self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib)
        self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.cython_wrapper.set_static_shape_map(self.static_shape_map)

        self._post_init()

    def _process_dynamic_symbolic(self) -> Dict[tir.Var, Tuple[int, int]]:
        """Extract information about dynamic shapes from the TIR function.
        
        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                        dynamic_symbolic_map[shape] = (i, j)
        return dynamic_symbolic_map

    def _process_buffer_dtype(self) -> Dict[tir.Var, Tuple[int, torch.dtype]]:
        """Extract information about buffer dtypes from the TIR function.
        
        Maps buffer variables to their corresponding dtypes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_dtype_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name, dtype = buffer.name, buffer.dtype
                buffer_dtype_map[name] = (i, map_torch_type(dtype))
        return buffer_dtype_map

    def _process_static_shape(self) -> Dict[tir.Var, List[Tuple[int, int]]]:
        """Extract information about static shapes from the TIR function.
        
        Maps buffer variables to their corresponding static shapes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        static_shape_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                shape = buffer.shape
                static_shape = []
                for j, s in enumerate(shape):
                    if isinstance(s, tir.IntImm):
                        static_shape.append((j, s.value))
                static_shape_map[name] = (i, static_shape)
        return static_shape_map

    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        """Low-level function to call the compiled CUDA kernel.
        
        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""

        def lambda_forward(*args, stream: int = -1):
            return self.cython_wrapper.forward([*args], stream=stream)

        return lambda_forward

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        """Returns the source path of the compiled library."""
        return self.lib_generator.srcpath

    @property
    def libpath(self):
        """Returns the path to the compiled library."""
        return self.lib_generator.libpath

    @property
    def lib_code(self):
        """Returns the code of the compiled library."""
        return self.lib_generator.lib_code

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        return (self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0)

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.mod.imported_modules[0].get_source()
        else:
            assert self.wrapped_source is not None, "Wrapped source is not available"
            return self.wrapped_source
