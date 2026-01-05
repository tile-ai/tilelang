"""The profiler and convert to torch utils"""

from __future__ import annotations
import ctypes
import logging
import torch

from typing import Callable, Any
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.param import KernelParam
from tvm import tir
from tvm.relax import TensorType

from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target, is_hip_target, is_cpu_target, is_metal_target
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.tensor import map_torch_type

logger = logging.getLogger(__name__)

try:
    from tilelang_cython_wrapper import CythonKernelWrapper
except ImportError:
    raise


def is_symbolic_expr(expr) -> bool:
    """Check if the expression is a symbolic expression.
    A symbolic expression can be a simple tvm.Var, or an tvm.PrimExpr containing tvm.Var.
    """
    return not isinstance(expr, tir.IntImm) and isinstance(expr, tir.PrimExpr)


class CythonKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using cython.

    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    lib: ctypes.CDLL | None = None  # Compiled library handle
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tir.Var, tuple[int, int]] | None = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: dict[int, str] | None = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: dict[tir.Var, tuple[int, torch.dtype]] | None = None
    # Maps buffer variables to their corresponding static shapes and strides,
    # e.g., {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape/strides = (16, 16)
    # }
    static_shape_map: dict[tir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    static_strides_map: dict[tir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    # Contains contiguous buffers
    static_contiguous_list: list[tir.Var] | None = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: dict[tir.Var, tuple[int, torch.device]] | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Initialize the adapter with the given TIR function or module.

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.device_kernel_source = device_kernel_source

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
        self.host_kernel_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.host_kernel_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()

        self.lib.get_last_error.restype = ctypes.c_char_p
        result = self.lib.init()
        if result != 0:
            error_msg = self.lib.get_last_error().decode("utf-8")
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
    def from_database(
        cls,
        params: list[TensorType],
        result_idx: list[int],
        target: str,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.host_kernel_source = host_kernel_source
        adapter.device_kernel_source = device_kernel_source
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
            error_msg = adapter.lib.get_last_error().decode("utf-8")
            raise RuntimeError(f"Initialization failed: {error_msg}")

        adapter.cython_wrapper = CythonKernelWrapper(adapter.result_idx, adapter.params, adapter.lib)
        adapter.cython_wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        adapter.cython_wrapper.set_buffer_dtype_map(adapter.buffer_dtype_map)
        adapter.cython_wrapper.set_static_shape_map(adapter.static_shape_map)
        adapter.cython_wrapper.set_static_strides_map(adapter.static_strides_map)
        adapter.cython_wrapper.set_static_contiguous_list(adapter.static_contiguous_list)
        adapter.cython_wrapper.set_buffer_device_map(adapter.buffer_device_map)
        adapter.cython_wrapper.set_ptr_map(adapter.ptr_map)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tir.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (1, i, j)
        return dynamic_symbolic_map

    def _process_buffer_dtype(self) -> dict[tir.Var, tuple[int, torch.dtype]]:
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

    def _process_ptr_map(self) -> dict[int, str]:
        """Extract information about pointer arguments from the TIR function.

        Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        ptr_map = {}
        for i, param in enumerate(params):
            if param.dtype == "handle":
                ptr_map[i] = param.name
        return ptr_map

    def _process_static_buffer_infos(
        self,
    ) -> tuple[dict[tir.Var, tuple[int, list[tuple[int, int]]]], dict[tir.Var, tuple[int, list[tuple[int, int]]]], list[tuple[tir.Var]]]:
        """Extract information about static shapes from the TIR function.

        Maps buffer variables to their corresponding static shapes.
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
                    elif is_symbolic_expr(s):
                        static_shape.append((j, -1))  # -1 for symbolic
                    else:
                        raise ValueError(f"Unsupported shape type: {type(s)}")
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

    def _process_buffer_device(self) -> dict[tir.Var, tuple[int, torch.device]]:
        """Extract information about buffer devices from the TIR function.

        Maps buffer variables to their corresponding devices.
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
        elif is_metal_target(self.target):
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported target: {self.target}")

        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                buffer_device_map[name] = (i, device)
        return buffer_device_map

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Low-level function to call the compiled CUDA kernel.

        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args]
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
            return self.cython_wrapper.forward([*args], stream=stream, skip_tensor_validation=skip_tensor_validation)

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
        return self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.device_kernel_source
        else:
            # Wrapper only has host kernel source
            assert self.host_kernel_source is not None, "Wrapped source is not available"
            return self.host_kernel_source

    def get_host_source(self):
        """Returns the source code of the host function."""
        return self.host_kernel_source
