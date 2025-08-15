"""The profiler and convert to torch utils"""

import ctypes
import fcntl
import hashlib
import logging
import site
import sys
import sysconfig
import torch
import os
from pathlib import Path

from typing import List, Optional, Union, Callable, Dict, Tuple, Any
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.param import KernelParam
from tvm import tir
from tvm.relax import TensorType

from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target, is_hip_target, is_cpu_target
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.tensor import map_torch_type
from tilelang.contrib.cc import get_cplus_compiler

logger = logging.getLogger(__name__)


def get_cython_compiler() -> Optional[str]:
    """Return the path to the Cython compiler.

    Returns
    -------
    out: Optional[str]
        The path to the Cython compiler, or None if none was found.
    """

    cython_names = ["cython", "cython3"]

    # Check system PATH
    dirs_in_path = list(os.get_exec_path())

    # Add user site-packages bin directory
    user_base = site.getuserbase()
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.exists(user_bin):
            dirs_in_path = [user_bin] + dirs_in_path

    # If in a virtual environment, add its bin directory
    if sys.prefix != sys.base_prefix:
        venv_bin = os.path.join(sys.prefix, "bin")
        if os.path.exists(venv_bin):
            dirs_in_path = [venv_bin] + dirs_in_path

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
    lock_file = cache_path.with_suffix('.lock')
    with open(lock_file, 'w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.exists():
                try:
                    if cache_path.stat().st_size > 1024:
                        return ctypes.CDLL(str(cache_path)), cache_path
                    else:
                        cache_path.unlink()  # remove the incomplete file
                except Exception as e:
                    logger.error(f"Failed to load cached library: {e}")
                    return None, cache_path
            return None, cache_path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    cache_path = cache_dir / f"{code_hash}.so"
    lock_file = cache_path.with_suffix('.lock')

    # Check if cached version exists and is valid
    need_compile = True
    if md5_path.exists() and library_path.exists():
        with open(md5_path, "r") as f:
            cached_hash = f.read().strip()
            if cached_hash == code_hash:
                logger.debug("Cython JIT adapter is up to date, no need to compile...")
                need_compile = False
            else:
                logger.info("Cython JIT adapter is out of date, need to recompile...")
    else:
        logger.info("No cached version found for Cython JIT adapter, need to compile...")

    if need_compile:
        logger.info("Waiting for lock to compile Cython JIT adapter...")
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                # After acquiring the lock, check again if the file has been compiled by another process
                if md5_path.exists() and library_path.exists():
                    with open(md5_path, "r") as f:
                        cached_hash = f.read().strip()
                        if cached_hash == code_hash:
                            logger.info(
                                "Another process has already compiled the file, using it...")
                            need_compile = False

                if need_compile:
                    logger.info("Compiling Cython JIT adapter...")
                    temp_path = cache_dir / f"temp_{code_hash}.so"

                    with open(md5_path, "w") as f:
                        f.write(code_hash)

                    # compile the cython_wrapper.pyx file into .cpp
                    cython = get_cython_compiler()
                    if cython is None:
                        raise Exception("Cython is not installed, please install it first.")
                    os.system(f"{cython} {cython_wrapper_path} --cplus -o {source_path}")
                    python_include_path = sysconfig.get_path("include")
                    cc = get_cplus_compiler()
                    command = f"{cc} -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I{python_include_path} {source_path} -o {temp_path}"
                    os.system(command)

                    # rename the temp file to the library file
                    temp_path.rename(library_path)
            except Exception as e:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                raise Exception(f"Failed to compile Cython JIT adapter: {e}") from e
            finally:
                if lock_file.exists():
                    lock_file.unlink()

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
    target: Union[str, Target] = "cuda"
    ir_module: Optional[tvm.IRModule] = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    kernel_global_source: Optional[str] = None
    lib: Optional[ctypes.CDLL] = None  # Compiled library handle
    wrapped_source: Optional[str] = None  # Generated C++ wrapper code
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: Optional[Dict[int, str]] = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: Optional[Dict[tir.Var, Tuple[int, torch.dtype]]] = None
    # Maps buffer variables to their corresponding static shapes and strides,
    # e.g., {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape/strides = (16, 16)
    # }
    static_shape_map: Optional[Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]]] = None
    static_strides_map: Optional[Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]]] = None
    # Contains contiguous buffers
    static_contiguous_list: Optional[List[tir.Var]] = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: Optional[Dict[tir.Var, Tuple[int, torch.device]]] = None
    # Pass configs for the compiler
    pass_configs: Optional[Dict[str, Any]] = None

    def __init__(self,
                 params: List[KernelParam],
                 result_idx: List[int],
                 target: Union[str, Target],
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 compile_flags: Optional[List[str]] = None):
        """
                 Create an adapter by compiling and loading a CUDA (or other target) kernel from a TIR function or module and preparing the Cython wrapper.
                 
                 This constructor:
                 - Normalizes result indices and target.
                 - Converts a PrimFunc to an IRModule if needed and records the module.
                 - Extracts runtime metadata (dynamic symbolic map, buffer dtypes, pointer indices, buffer devices) and static buffer info (static shapes, static strides, statically contiguous buffers).
                 - Builds a TLWrapper and LibraryGenerator, produces wrapped kernel source, compiles a shared library, loads it, and initializes the library.
                 - Instantiates and configures a CythonKernelWrapper with the computed maps so the adapter is ready for runtime invocation.
                 
                 Args:
                     params: List of kernel parameters (input/output buffer descriptors).
                     result_idx: Indices of output tensors produced by the kernel; will be normalized.
                     target: Target specification (string or Target) used for lowering/compilation (e.g., "cuda").
                     func_or_mod: Primary TIR function (tir.PrimFunc) or a tvm.IRModule containing the kernel.
                     host_mod: Optional host-side IRModule to include during wrapping.
                     device_mod: Optional device-side IRModule to include during wrapping.
                     kernel_global_source: Optional global source code to include with the kernel.
                     verbose: If True, enable more verbose compilation/logging from the library generator.
                     pass_configs: Optional pass configuration dictionary forwarded to the wrapper and library generator.
                     compile_flags: Optional list of compiler flags forwarded to the library generator.
                 
                 Raises:
                     RuntimeError: If the compiled library's init() returns non-zero; the library's last error and generated source are appended to the message.
                 """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()

        static_buffer_infos = self._process_static_buffer_infos()
        self.static_shape_map = static_buffer_infos[0]
        self.static_strides_map = static_buffer_infos[1]
        self.static_contiguous_list = static_buffer_infos[2]

        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target, verbose=verbose)
        self.lib_generator.assign_pass_configs(pass_configs)
        self.lib_generator.assign_compile_flags(compile_flags)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()

        self.lib.get_last_error.restype = ctypes.c_char_p
        result = self.lib.init()
        if result != 0:
            error_msg = self.lib.get_last_error().decode('utf-8')
            error_msg += f"\n{self.lib_code}"
            raise RuntimeError(f"Initialization failed: {error_msg}")

        self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib)
        self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.cython_wrapper.set_static_shape_map(self.static_shape_map)
        self.cython_wrapper.set_static_strides_map(self.static_strides_map)
        self.cython_wrapper.set_static_contiguous_list(self.static_contiguous_list)
        self.cython_wrapper.set_buffer_device_map(self.buffer_device_map)
        self.cython_wrapper.set_ptr_map(self.ptr_map)
        self._post_init()

    @classmethod
    def from_database(cls,
                      params: List[TensorType],
                      result_idx: List[int],
                      target: str,
                      func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                      kernel_global_source: str,
                      kernel_lib_path: str,
                      verbose: bool = False,
                      pass_configs: Optional[Dict[str, Any]] = None,
                      compile_flags: Optional[List[str]] = None):
        """
                      Create a CythonKernelAdapter from a prebuilt kernel library.
                      
                      This alternate constructor builds an adapter without compiling: it loads the given IR function or module,
                      reads static/dynamic metadata from the TIR, loads and initializes the prebuilt shared library, and
                      configures a CythonKernelWrapper with the derived maps (dynamic-symbolic, dtype, pointer, device,
                      static shapes/strides/contiguity).
                      
                      Parameters:
                          params: List of kernel parameter TensorType objects used to derive indices, dtypes, and pointer info.
                          result_idx: Indices of kernel return values to expose (will be normalized).
                          target: Target identifier (e.g., "cuda") used to determine devices and codegen target.
                          func_or_mod: A tir.PrimFunc or tvm.IRModule containing the kernel; used to extract symbolic and static metadata.
                          kernel_global_source: Kernel source string (used as the wrapped_source in the adapter).
                          kernel_lib_path: Filesystem path to the prebuilt shared library to load.
                          verbose: If True, enables more verbose internal operations during loading/initialization.
                          pass_configs: Optional pass configuration dictionary passed to the library generator.
                          compile_flags: Optional list of compile flags (kept for API compatibility; not used to build when loading).
                      
                      Returns:
                          An initialized CythonKernelAdapter instance wired to the loaded library and ready for use.
                      
                      Raises:
                          RuntimeError: If the loaded library's init() returns non-zero; the library's last error message is included.
                      """
                      adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.kernel_global_source = kernel_global_source
        adapter.wrapped_source = kernel_global_source
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target.canon_target(determine_target(target))

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter.buffer_dtype_map = adapter._process_buffer_dtype()
        adapter.ptr_map = adapter._process_ptr_map()
        adapter.buffer_device_map = adapter._process_buffer_device()

        static_buffer_infos = adapter._process_static_buffer_infos()
        adapter.static_shape_map = static_buffer_infos[0]
        adapter.static_strides_map = static_buffer_infos[1]
        adapter.static_contiguous_list = static_buffer_infos[2]

        adapter.verbose = verbose
        adapter.lib_generator = LibraryGenerator(adapter.target, verbose=verbose)
        adapter.lib_generator.assign_pass_configs(pass_configs)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib = adapter.lib_generator.load_lib(lib_path=kernel_lib_path)

        adapter.lib.get_last_error.restype = ctypes.c_char_p
        result = adapter.lib.init()
        if result != 0:
            error_msg = adapter.lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Initialization failed: {error_msg}")

        adapter.cython_wrapper = CythonKernelWrapper(adapter.result_idx, adapter.params,
                                                     adapter.lib)
        adapter.cython_wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        adapter.cython_wrapper.set_buffer_dtype_map(adapter.buffer_dtype_map)
        adapter.cython_wrapper.set_static_shape_map(adapter.static_shape_map)
        adapter.cython_wrapper.set_static_strides_map(adapter.static_strides_map)
        adapter.cython_wrapper.set_static_contiguous_list(adapter.static_contiguous_list)
        adapter.cython_wrapper.set_buffer_device_map(adapter.buffer_device_map)
        adapter.cython_wrapper.set_ptr_map(adapter.ptr_map)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> Dict[tir.Var, Tuple[int, int]]:
        """
        Map dynamic TIR symbols referenced in the primary PrimFunc to runtime resolution info.
        
        Scans the function's buffer shapes and strides and returns a mapping from each dynamic
        symbol (a tir.Var appearing in a shape or stride) to a 3-tuple (kind, param_index, dim_index),
        where `kind` is 0 for a shape symbol and 1 for a stride symbol, `param_index` is the
        index of the buffer parameter that uses the symbol, and `dim_index` is the dimension index
        within that buffer. Symbols that are function parameters or already recorded are ignored.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if (isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and
                        (shape not in params)):
                        dynamic_symbolic_map[shape] = (0, i, j)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, stride in enumerate(buffer.strides):
                    if (isinstance(stride, tir.Var) and (stride not in dynamic_symbolic_map) and
                        (stride not in params)):
                        dynamic_symbolic_map[stride] = (1, i, j)
        return dynamic_symbolic_map

    def _process_buffer_dtype(self) -> Dict[tir.Var, Tuple[int, torch.dtype]]:
        """
        Return a mapping of buffer names to their parameter index and corresponding PyTorch dtype.
        
        Scans the primary TIR function's parameters and its buffer_map. For each parameter that is a buffer, records an entry keyed by the buffer's name (str) with value (parameter_index, torch.dtype) where parameter_index is the index of that parameter in the function signature and the dtype is converted to a PyTorch dtype via map_torch_type.
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

    def _process_ptr_map(self) -> Dict[int, str]:
        """
        Return a mapping of function parameter indices to pointer parameter names.
        
        Scans the primary TIR function's parameters and records those with dtype 'handle'
        (as pointer arguments). The returned dict maps the parameter index (int) to the
        parameter's name (str).
        """
        func = self.prim_func
        params = func.params
        ptr_map = {}
        for i, param in enumerate(params):
            if param.dtype == 'handle':
                ptr_map[i] = param.name
        return ptr_map

    def _process_static_buffer_infos(self) -> \
            Tuple[Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]],
                  Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]],
                  List[Tuple[tir.Var]]]:
        """
            Collect static shape and stride metadata for buffers referenced by the primary TIR function.
            
            Scans the primary TIR PrimFunc's parameters and buffer_map, extracting:
            - static sizes from dimensions that are literal integers (tir.IntImm),
            - static strides from stride entries that are tir.IntImm,
            - whether a buffer is statically contiguous by checking if each stride equals the product of following dimensions.
            
            Returns:
                Tuple containing three elements:
                - static_shape_map: Dict mapping buffer variable -> (param_index, List[(dim_index, size)]),
                  where each list entry is a (dimension_index, static_size) for dimensions that are integer immediates.
                - static_strides_map: Dict mapping buffer variable -> (param_index, List[(dim_index, stride)]),
                  where each list entry is a (dimension_index, static_stride) for strides that are integer immediates.
                - static_contiguous_list: List of tuples (param_index, buffer_name) for buffers determined to be statically contiguous.
            """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        static_shape_map = {}
        static_strides_map = {}
        static_contiguous_list = list()
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                static_shape, static_strides = [], []
                for j, s in enumerate(buffer.shape):
                    if isinstance(s, tir.IntImm):
                        static_shape.append((j, s.value))
                for j, s in enumerate(buffer.strides):
                    if isinstance(s, tir.IntImm):
                        static_strides.append((j, s.value))
                is_contiguous, prod = True, 1
                for dim, stride in reversed(list(zip(buffer.shape, buffer.strides))):
                    is_contiguous &= bool(stride == prod)
                    prod *= dim
                static_shape_map[buffer.name] = (i, static_shape)
                static_strides_map[buffer.name] = (i, static_strides)
                if is_contiguous:
                    static_contiguous_list.append((i, buffer.name))
        return static_shape_map, static_strides_map, static_contiguous_list

    def _process_buffer_device(self) -> Dict[tir.Var, Tuple[int, torch.device]]:
        """
        Return a mapping from buffer name to the device and its parameter index.
        
        Determines the torch.device for all buffers in the primary TIR function based on the adapter's target:
        - CUDA/HIP targets -> torch.device("cuda")
        - CPU target       -> torch.device("cpu")
        
        Returns:
            Dict[str, Tuple[int, torch.device]]: Mapping where each key is the buffer's name and the value is a tuple
            (parameter_index, torch.device).
        
        Raises:
            ValueError: If the adapter target is unsupported.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_device_map = {}
        device = None
        if is_cuda_target(self.target) or is_hip_target(self.target):
            device = torch.device("cuda")
        elif is_cpu_target(self.target):
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported target: {self.target}")

        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                buffer_device_map[name] = (i, device)
        return buffer_device_map

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

        def lambda_forward(*args, stream: int = -1, skip_tensor_validation: bool = False):
            """
            Args:
                args: List of input tensors
                stream: CUDA stream ID, default to -1, will use the current stream if not specified
                skip_tensor_validation: Whether to skip tensor attributes validation which
                includes shape, dtype, device, etc.
            """
            return self.cython_wrapper.forward([*args],
                                               stream=stream,
                                               skip_tensor_validation=skip_tensor_validation)

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
        """
        Return True if the kernel contains any dynamic shape or stride symbols.
        
        Checks whether the adapter has a non-empty dynamic_symbolic_map that indicates
        the kernel exposes dynamic dimensions/strides that must be supplied at runtime.
        
        Returns:
            bool: True when dynamic_symbolic_map is present and contains entries.
        """
        return self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0

    def get_kernel_source(self, kernel_only: bool = False):
        """
        Return the kernel source.
        
        If `kernel_only` is True, returns the raw kernel global source (may be None if not set).
        If `kernel_only` is False, returns the wrapped source prepared for the JIT wrapper; raises
        AssertionError if the wrapped source is not available.
        
        Parameters:
            kernel_only (bool): If True, return only the kernel global source; otherwise return the wrapped source.
        
        Returns:
            str | None: The requested source string, or None when `kernel_only` is True and no kernel_global_source is set.
        
        Raises:
            AssertionError: If `kernel_only` is False and the wrapped source has not been produced.
        """
        if kernel_only:
            return self.kernel_global_source
        else:
            assert self.wrapped_source is not None, "Wrapped source is not available"
            return self.wrapped_source
