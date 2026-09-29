"""The profiler and convert to torch utils"""

from __future__ import annotations
import ctypes
import logging
import torch

from typing import Any
from collections.abc import Callable
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.param import KernelParam
from tvm import arith, tirx
from tvm.relax import TensorType

from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_ascend_target, is_cpu_target, is_cuda_target, is_hip_target, is_metal_target
from tilelang.backend.target import determine_target
from tilelang.utils.language import prim_expr_equal, retrieve_func_from_module

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
    return not isinstance(expr, tirx.IntImm) and isinstance(expr, tirx.PrimExpr)


def _storage_pack_factor(dtype: tvm.DataType, target: Target) -> int:
    """Return the target-specific host-storage packing factor for ``dtype``."""
    # Keep the Cython adapter on the same ABI definition as KernelParam.  In
    # particular, bool is one byte in Torch even though its TIR dtype has one
    # logical bit, while packed int4/FP4 storage is only enabled for PTO.
    return KernelParam(dtype, []).storage_packing_factor(target=target)


def _accepted_storage_dtypes(dtype: tvm.DataType, target: Target) -> torch.dtype | tuple[torch.dtype, ...]:
    torch_dtype = dtype.as_torch()
    if _storage_pack_factor(dtype, target) > 1 and torch_dtype != torch.int8:
        return (torch_dtype, torch.int8)
    return torch_dtype


def _pack_static_value(value: int, pack_factor: int, what: str, keep_unit: bool = False) -> int:
    if pack_factor == 1 or (keep_unit and value == 1):
        return value
    if value % pack_factor != 0:
        raise ValueError(f"{what} must be divisible by storage pack factor {pack_factor}, got {value}")
    return value // pack_factor


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
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tirx.Var, tuple[int, int, int, int]] | None = None
    # Maps scalar params to their TIR vars, None elsewhere
    scalar_param_vars: list[tirx.Var | None] | None = None
    # Maps symbolic variable names to ALL buffers that carry them, for cascaded None resolution
    dynamic_symbolic_sources: dict[str, list[tuple[int, int, int]]] | None = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: dict[int, str] | None = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: dict[tirx.Var, tuple[int, torch.dtype | tuple[torch.dtype, ...]]] | None = None
    # Per-parameter storage metadata used when the Python wrapper allocates outputs.
    # The shape in KernelParam is logical, while PyTorch tensors for sub-byte dtypes
    # are allocated in packed storage elements.
    param_storage_metadata: list[tuple[int, int]] | None = None
    # Maps buffer variables to their corresponding static shapes and strides,
    # e.g., {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape/strides = (16, 16)
    # }
    static_shape_map: dict[tirx.Var, tuple[int, list[tuple[int, int]]]] | None = None
    static_strides_map: dict[tirx.Var, tuple[int, list[tuple[int, int]]]] | None = None
    # Contains contiguous buffers
    static_contiguous_list: list[tirx.Var] | None = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: dict[tirx.Var, tuple[int, torch.device]] | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
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

        if isinstance(func_or_mod, tirx.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target(determine_target(target))

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.dynamic_symbolic_sources = self._process_dynamic_symbolic_sources()
        self.scalar_param_vars = self._process_scalar_param_vars()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.param_storage_metadata = self._process_param_storage_metadata()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()

        static_buffer_infos = self._process_static_buffer_infos()
        self.static_shape_map = static_buffer_infos[0]
        self.static_strides_map = static_buffer_infos[1]
        self.static_contiguous_list = static_buffer_infos[2]
        self.dynamic_strides_map = static_buffer_infos[3]

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
        if self.wrapper.pto_kernel_source is not None:
            self.lib_generator.update_pto_kernels(self.wrapper.pto_kernel_source, self.wrapper.pto_kernel_names)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()

        self.lib.get_last_error.restype = ctypes.c_char_p
        result = self.lib.init()
        if result != 0:
            error_msg = self.lib.get_last_error().decode("utf-8")
            error_msg += f"\n{self.lib_code}"
            raise RuntimeError(f"Initialization failed: {error_msg}")

        self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib, *_device_providers())
        self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.cython_wrapper.set_dynamic_symbolic_sources(self.dynamic_symbolic_sources)
        self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.cython_wrapper.set_param_storage_metadata(self.param_storage_metadata)
        self.cython_wrapper.set_static_shape_map(self.static_shape_map)
        self.cython_wrapper.set_static_strides_map(self.static_strides_map)
        self.cython_wrapper.set_dynamic_strides_map(self.dynamic_strides_map)
        self.cython_wrapper.set_scalar_param_vars(self.scalar_param_vars)
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
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
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

        if isinstance(func_or_mod, tirx.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target(determine_target(target))

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter.dynamic_symbolic_sources = adapter._process_dynamic_symbolic_sources()
        adapter.scalar_param_vars = adapter._process_scalar_param_vars()
        adapter.buffer_dtype_map = adapter._process_buffer_dtype()
        adapter.param_storage_metadata = adapter._process_param_storage_metadata()
        adapter.ptr_map = adapter._process_ptr_map()
        adapter.buffer_device_map = adapter._process_buffer_device()

        static_buffer_infos = adapter._process_static_buffer_infos()
        adapter.static_shape_map = static_buffer_infos[0]
        adapter.static_strides_map = static_buffer_infos[1]
        adapter.static_contiguous_list = static_buffer_infos[2]
        adapter.dynamic_strides_map = static_buffer_infos[3]

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
        adapter.cython_wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        adapter.cython_wrapper.set_dynamic_symbolic_sources(adapter.dynamic_symbolic_sources)
        adapter.cython_wrapper.set_buffer_dtype_map(adapter.buffer_dtype_map)
        adapter.cython_wrapper.set_param_storage_metadata(adapter.param_storage_metadata)
        adapter.cython_wrapper.set_static_shape_map(adapter.static_shape_map)
        adapter.cython_wrapper.set_static_strides_map(adapter.static_strides_map)
        adapter.cython_wrapper.set_dynamic_strides_map(adapter.dynamic_strides_map)
        adapter.cython_wrapper.set_scalar_param_vars(adapter.scalar_param_vars)
        adapter.cython_wrapper.set_static_contiguous_list(adapter.static_contiguous_list)
        adapter.cython_wrapper.set_buffer_device_map(adapter.buffer_device_map)
        adapter.cython_wrapper.set_ptr_map(adapter.ptr_map)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> dict[tirx.Var, tuple[int, int, int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension, storage_scale)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride.
        storage_scale compensates for sub-byte dtypes (e.g. float4_e2m1fn) where torch
        reports packed storage units but the kernel expects logical element units.
        """

        def shape_scale(buffer, dim_idx: int) -> int:
            return _storage_pack_factor(buffer.dtype, self.target) if dim_idx == len(buffer.shape) - 1 else 1

        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tirx.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j, shape_scale(buffer, j))
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                stride_scale = _storage_pack_factor(buffer.dtype, self.target)
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tirx.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (1, i, j, stride_scale)
        return dynamic_symbolic_map

    def _process_dynamic_symbolic_sources(self) -> dict[str, list[tuple[int, int, int]]]:
        """Build a multi-source map for cascaded None resolution.

        For each dynamic symbol, maps to ALL buffers that carry it as (buffer_idx, dim_idx, storage_scale).
        This allows the Cython wrapper to find a non-None carrier when some buffers are None.
        """

        def shape_scale(buffer, dim_idx: int) -> int:
            return _storage_pack_factor(buffer.dtype, self.target) if dim_idx == len(buffer.shape) - 1 else 1

        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        sources: dict[str, list[tuple[int, int, int]]] = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                stride_scale = _storage_pack_factor(buffer.dtype, self.target)
                for j, dim in enumerate(buffer.shape):
                    if isinstance(dim, tirx.Var) and dim not in params:
                        key = str(dim)
                        if key not in sources:
                            sources[key] = []
                        sources[key].append((i, j, shape_scale(buffer, j)))
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tirx.Var) and stride not in params:
                        key = str(stride)
                        if key not in sources:
                            sources[key] = []
                        sources[key].append((i, j, stride_scale))
        return sources

    def _process_scalar_param_vars(self) -> list[tirx.Var | None]:
        """Map scalar params to their TIR vars, None elsewhere.

        Stride expressions may reference explicit scalar params, which are
        excluded from dynamic_symbolic_map since the caller passes their
        values directly at launch.
        """
        func = self.prim_func
        buffer_map = func.buffer_map
        scalar_param_vars: list[tirx.Var | None] = []
        for param in func.params:
            if param not in buffer_map and param.dtype != "handle":
                scalar_param_vars.append(param)
            else:
                scalar_param_vars.append(None)
        return scalar_param_vars

    def _process_buffer_dtype(self) -> dict[tirx.Var, tuple[int, torch.dtype | tuple[torch.dtype, ...]]]:
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
                buffer_dtype_map[name] = (i, _accepted_storage_dtypes(dtype, self.target))
        return buffer_dtype_map

    def _process_param_storage_metadata(self) -> list[tuple[int, int]]:
        """Return (packed_dim, pack_factor) for each parameter.

        KernelParam shapes remain in logical element units. For sub-byte element
        types, PyTorch represents the backing tensor with packed storage elements,
        so wrapper-created outputs must divide the packed dimension before calling
        torch.empty.
        """
        metadata = []
        for param in self.params:
            pack_factor = param.storage_packing_factor(target=self.target)
            if pack_factor > 1 and len(param.shape) > 0:
                metadata.append((len(param.shape) - 1, pack_factor))
            else:
                metadata.append((-1, 1))
        return metadata

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
    ) -> tuple[
        dict[str, tuple[int, list[tuple[int, int]]]],
        dict[str, tuple[int, list[tuple[int, int]]]],
        list[tuple[int, str]],
        dict[str, tuple[int, list[tuple[int, tirx.PrimExpr, int]]]],
    ]:
        """Extract information about static shapes from the TIR function.

        Maps buffer variables to their corresponding static shapes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        static_shape_map = {}
        static_strides_map = {}
        dynamic_strides_map = {}
        static_contiguous_list = list()
        analyzer = arith.Analyzer()
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                static_shape, static_strides, dynamic_strides = [], [], []
                packing_factor = KernelParam.from_buffer(buffer).storage_packing_factor(target=self.target)
                innermost_dim = len(buffer.shape) - 1
                for j, s in enumerate(buffer.shape):
                    if isinstance(s, tirx.IntImm):
                        extent = s.value
                        if j == innermost_dim and packing_factor > 1:
                            if extent % packing_factor:
                                raise ValueError(
                                    f"The innermost dimension of {buffer.dtype} must be divisible by "
                                    f"its packing factor ({packing_factor}), got {extent}"
                                )
                            extent //= packing_factor
                        static_shape.append((j, extent))
                    elif is_symbolic_expr(s):
                        static_shape.append((j, -1))  # -1 for symbolic
                    else:
                        raise ValueError(f"Unsupported shape type: {type(s)}")
                for j, s in enumerate(buffer.strides):
                    if j != innermost_dim and packing_factor > 1 and is_symbolic_expr(s):
                        dynamic_strides.append((j, s, packing_factor))
                    if j == innermost_dim and packing_factor > 1 and (not isinstance(s, tirx.IntImm) or s.value != 1):
                        raise ValueError(f"Packed {buffer.dtype} buffers require a static innermost stride of 1")
                    if isinstance(s, tirx.IntImm):
                        stride = s.value
                        if j != innermost_dim and packing_factor > 1:
                            if stride % packing_factor:
                                raise ValueError(
                                    f"The stride of packed {buffer.dtype} must be divisible by "
                                    f"its packing factor ({packing_factor}), got {stride}"
                                )
                            stride //= packing_factor
                        static_strides.append((j, stride))
                is_contiguous, prod = True, 1
                for dim, stride in reversed(list(zip(buffer.shape, buffer.strides))):
                    if not (prim_expr_equal(stride, prod) or analyzer.can_prove_equal(stride, prod)):
                        is_contiguous = False
                        break
                    prod *= dim
                static_shape_map[buffer.name] = (i, static_shape)
                static_strides_map[buffer.name] = (i, static_strides)
                if dynamic_strides:
                    dynamic_strides_map[buffer.name] = (i, dynamic_strides)
                if is_contiguous:
                    static_contiguous_list.append((i, buffer.name))
        return static_shape_map, static_strides_map, static_contiguous_list, dynamic_strides_map

    def _process_buffer_device(self) -> dict[tirx.Var, tuple[int, torch.device]]:
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
        elif is_ascend_target(self.target):
            device = torch.device("npu")
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
    def prim_func(self) -> tirx.PrimFunc:
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
