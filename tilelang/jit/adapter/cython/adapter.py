"""The profiler and convert to torch utils"""

from __future__ import annotations
import ctypes
import logging
import torch

from typing import Any
from collections.abc import Callable
from tilelang import tvm as tvm
from tvm_tirx.target import Target
from tilelang.engine.param import KernelParam
from tvm_tirx import ir, tirx
from tilelang.jit.adapter.abi import source_abi_plan

from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_ascend_target, is_cpu_target, is_cuda_target, is_hip_target, is_metal_target
from tilelang.backend.target import determine_target
from tilelang.utils.language import retrieve_func_from_module

logger = logging.getLogger(__name__)

try:
    from tilelang_cython_wrapper import CythonKernelWrapper
except ImportError:
    raise


def _device_providers():
    """Resolve Torch devices and their current streams lazily at runtime."""
    current_device_functor = None
    current_stream_functors = {}

    def current_device():
        nonlocal current_device_functor
        if current_device_functor is None:
            current_device_functor = BaseKernelAdapter.get_current_device_functor()
        return current_device_functor()

    def current_stream(device: torch.device):
        device_key = (device.type, device.index)
        stream_functor = current_stream_functors.get(device_key)
        if stream_functor is None:
            stream_functor = BaseKernelAdapter.get_current_stream_functor(device)
            current_stream_functors[device_key] = stream_functor
        return stream_functor()

    return current_device, current_stream


def is_symbolic_expr(expr) -> bool:
    """Check if the expression is a symbolic expression.
    A symbolic expression can be a simple tvm.Var, or an tvm.PrimExpr containing tvm.Var.
    """
    return not isinstance(expr, ir.IntImm) and ir.is_prim_expr(expr)


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
    kernel_global_source: str | None = None  # Alias for device_kernel_source for compatibility
    lib: ctypes.CDLL | None = None  # Compiled library handle
    # Maps symbolic variables to (kind, buffer_index, dimension, stride_scale)
    dynamic_symbolic_map: dict[ir.Var, tuple[int, int, int, int]] | None = None
    # Maps symbolic variable names to ALL buffers that carry them, for cascaded None resolution
    dynamic_symbolic_sources: dict[str, list[tuple[int, int, int, int]]] | None = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: dict[int, str] | None = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: dict[ir.Var, tuple[int, torch.dtype]] | None = None
    # Maps buffer variables to their corresponding static shapes and strides,
    # e.g., {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape/strides = (16, 16)
    # }
    static_shape_map: dict[ir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    static_strides_map: dict[ir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    # Contains contiguous buffers
    static_contiguous_list: list[ir.Var] | None = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: dict[ir.Var, tuple[int, torch.device]] | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.Function | tvm.IRModule,
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
        self.kernel_global_source = device_kernel_source  # Set alias for compatibility

        if isinstance(func_or_mod, tirx.Function):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.params = source_abi_plan(self).validate_parameter_metadata(self.params)
        self.target = Target(determine_target(target))

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.dynamic_symbolic_sources = self._process_dynamic_symbolic_sources()
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

        self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib, *_device_providers())
        self.cython_wrapper.set_source_abi_plan(source_abi_plan(self))
        self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.cython_wrapper.set_dynamic_symbolic_sources(self.dynamic_symbolic_sources)
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
        params: list[KernelParam],
        result_idx: list[int],
        target: str,
        func_or_mod: tirx.Function | tvm.IRModule,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        device_kernel_source = adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        adapter.kernel_global_source = device_kernel_source.text  # Set alias for compatibility
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tirx.Function):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        adapter.params = source_abi_plan(adapter).validate_parameter_metadata(adapter.params)
        target = determine_target(target, return_object=True)
        adapter.target = Target(determine_target(target))

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter.dynamic_symbolic_sources = adapter._process_dynamic_symbolic_sources()
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

        adapter.cython_wrapper = CythonKernelWrapper(adapter.result_idx, adapter.params, adapter.lib, *_device_providers())
        adapter.cython_wrapper.set_source_abi_plan(source_abi_plan(adapter))
        adapter.cython_wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        adapter.cython_wrapper.set_dynamic_symbolic_sources(adapter.dynamic_symbolic_sources)
        adapter.cython_wrapper.set_buffer_dtype_map(adapter.buffer_dtype_map)
        adapter.cython_wrapper.set_static_shape_map(adapter.static_shape_map)
        adapter.cython_wrapper.set_static_strides_map(adapter.static_strides_map)
        adapter.cython_wrapper.set_static_contiguous_list(adapter.static_contiguous_list)
        adapter.cython_wrapper.set_buffer_device_map(adapter.buffer_device_map)
        adapter.cython_wrapper.set_ptr_map(adapter.ptr_map)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> dict[ir.Var, tuple[int, int, int, int]]:
        plan = source_abi_plan(self)
        preferred = plan.primary_sources(self.result_idx)
        return {variable: preferred[variable].tuple() if variable in preferred else (-1, -1, -1, 1) for variable in plan.implicit_symbols}

    def _process_dynamic_symbolic_sources(self):
        plan = source_abi_plan(self)
        return {variable: [source.tuple() for source in sources if source.kind != 2] for variable, sources in plan.sources.items()}

    def _process_buffer_dtype(self):
        plan = source_abi_plan(self)
        return {tensor.variable: (index, plan.parameters[index].torch_dtype()) for index, tensor in plan.tensors.items()}

    def _process_ptr_map(self) -> dict[int, str]:
        # TensorType carries shape and element metadata, but the raw source ABI
        # still passes its storage address.  T.make_tensor may refine a T.ptr
        # parameter into TensorType, including when callers pass data_ptr().
        return {
            index: parameter.name
            for index, parameter in enumerate(self.prim_func.params)
            if isinstance(parameter.ty, (ir.PtrType, tirx.TensorType))
        }

    def _process_static_buffer_infos(self):
        plan = source_abi_plan(self)
        shapes, strides, contiguous = {}, {}, []
        for index, tensor in plan.tensors.items():
            shapes[tensor.variable] = (
                index,
                [(axis, int(value.value) if isinstance(value, ir.IntImm) else -1) for axis, value in enumerate(tensor.shape)],
            )
            strides[tensor.variable] = (
                index,
                [(axis, int(value.value)) for axis, value in enumerate(tensor.strides) if isinstance(value, ir.IntImm)],
            )
            # Runtime shape/stride equalities use the complete affine layout.
            # A compact layout constructed by TensorType is already represented
            # by these strides, including singleton axes; no name-based guess
            # or global Analyzer proof is needed to select an extra check.
        return shapes, strides, contiguous

    def _process_buffer_device(self):
        if is_cuda_target(self.target) or is_hip_target(self.target):
            device = torch.device("cuda")
        elif is_ascend_target(self.target):
            device = torch.device("npu")
        elif is_cpu_target(self.target):
            device = torch.device("cpu")
        elif is_metal_target(self.target):
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported target: {self.target}")
        return {tensor.variable: (index, device) for index, tensor in source_abi_plan(self).tensors.items()}

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
    def prim_func(self) -> tirx.Function:
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
            source = self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
            if source is not None:
                self.kernel_global_source = source
            return source
        else:
            # Wrapper only has host kernel source
            source = self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
            assert source is not None, "Wrapped source is not available"
            return source

    def get_host_source(self):
        """Returns the source code of the host function."""
        return self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
