"""Compile and execute PTODSL kernels using the PTO launch wrapper."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import torch
from tvm import arith, tirx
from tvm.target import Target
from tilelang_pto_wrapper import PTOKernelWrapper

from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.utils.language import prim_expr_equal, retrieve_func_from_module

from .libgen import PTOLibraryGenerator
from .wrapper import TLPTOWrapper


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


def _storage_pack_factor(dtype: tvm.DataType) -> int:
    """Return the number of PTO logical values stored in one Torch element."""
    # Bool uses one byte per value. Only int4/uint4 and FP4 use packed storage.
    dtype_name = str(dtype).removeprefix("torch.")
    if not (dtype_name.startswith("float4") or dtype_name in {"int4", "uint4"}):
        return 1
    logical_bits = dtype.bits * dtype.lanes
    if logical_bits >= 8:
        return 1
    storage_bits = 8
    if storage_bits % logical_bits:
        raise ValueError(f"Cannot represent {dtype} in an {storage_bits}-bit Torch storage element")
    return storage_bits // logical_bits


def _accepted_storage_dtypes(dtype: tvm.DataType) -> torch.dtype | tuple[torch.dtype, ...]:
    torch_dtype = dtype.as_torch()
    if _storage_pack_factor(dtype) > 1 and torch_dtype != torch.int8:
        return (torch_dtype, torch.int8)
    return torch_dtype


class PTOKernelAdapter(BaseKernelAdapter):
    """Own PTO compilation and cached-library loading with a Cython launch ABI."""

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int | None,
        target: str | dict[str, object] | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        self._initialize(params, result_idx, target, func_or_mod, verbose, pass_configs, compile_flags)
        self.device_kernel_source = device_kernel_source
        self.kernel_global_source = device_kernel_source

        self.wrapper = TLPTOWrapper(self.target)
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.host_kernel_source = self.wrapper.wrap(device_kernel_source)

        source_wrapper = self.wrapper.source_wrapper
        self.lib_generator.update_lib_code(self.host_kernel_source)
        self.lib_generator.update_pto_kernels(source_wrapper.pto_kernel_source, source_wrapper.pto_kernel_names)
        self.lib_generator.compile_lib()
        self._load_library()

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int] | int | None,
        target: str | dict[str, object] | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Load the compiled PTO library without regenerating device or host code."""
        adapter = cls.__new__(cls)
        adapter._initialize(params, result_idx, target, func_or_mod, verbose, pass_configs, compile_flags)
        adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        device_kernel_source = adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        adapter.kernel_global_source = device_kernel_source.text
        adapter._load_library(kernel_lib_path)
        return adapter

    def _initialize(self, params, result_idx, target, func_or_mod, verbose, pass_configs, compile_flags):
        """Prepare the same argument metadata for fresh and cached kernels."""
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        if isinstance(func_or_mod, tirx.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
        self.target = Target(determine_target(target))
        self.verbose = verbose
        self.pass_configs = pass_configs

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.dynamic_symbolic_sources = self._process_dynamic_symbolic_sources()
        self.scalar_param_vars = self._process_scalar_param_vars()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.param_storage_metadata = self._process_param_storage_metadata()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()
        (
            self.static_shape_map,
            self.static_strides_map,
            self.static_contiguous_list,
            self.dynamic_strides_map,
        ) = self._process_static_buffer_infos()

        self.lib_generator = PTOLibraryGenerator(self.target, verbose=verbose)
        self.lib_generator.assign_pass_configs(pass_configs)
        self.lib_generator.assign_compile_flags(compile_flags)

    def _load_library(self, lib_path: str | None = None):
        self.lib = self.lib_generator.load_lib(lib_path=lib_path)
        self.lib.get_last_error.restype = ctypes.c_char_p
        if self.lib.init() != 0:
            error_msg = self.lib.get_last_error().decode("utf-8")
            if lib_path is None:
                error_msg += f"\n{self.lib_code}"
            raise RuntimeError(f"Initialization failed: {error_msg}")

        self.runtime_wrapper = PTOKernelWrapper(self.result_idx, self.params, self.lib, *_device_providers())
        self.runtime_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.runtime_wrapper.set_dynamic_symbolic_sources(self.dynamic_symbolic_sources)
        self.runtime_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.runtime_wrapper.set_param_storage_metadata(self.param_storage_metadata)
        self.runtime_wrapper.set_static_shape_map(self.static_shape_map)
        self.runtime_wrapper.set_static_strides_map(self.static_strides_map)
        self.runtime_wrapper.set_dynamic_strides_map(self.dynamic_strides_map)
        self.runtime_wrapper.set_scalar_param_vars(self.scalar_param_vars)
        self.runtime_wrapper.set_static_contiguous_list(self.static_contiguous_list)
        self.runtime_wrapper.set_buffer_device_map(self.buffer_device_map)
        self.runtime_wrapper.set_ptr_map(self.ptr_map)
        self._post_init()

    def _process_dynamic_symbolic(self) -> dict[tirx.Var, tuple[int, int, int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension, storage_scale)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride.
        storage_scale compensates for sub-byte dtypes (e.g. float4_e2m1fn) where torch
        reports packed storage units but the kernel expects logical element units.
        """

        def shape_scale(buffer, dim_idx: int) -> int:
            return _storage_pack_factor(buffer.dtype) if dim_idx == len(buffer.shape) - 1 else 1

        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        # Inputs are visited first. An output's shape is resolved from these entries
        # while that output is being allocated, so a dimension mentioned by both an
        # input and an output must be owned by the input; owning it on the output
        # would make the allocation loop read a slot it has not filled yet.
        ordered = [i for i in range(len(params)) if i not in self.result_idx]
        ordered += [i for i in range(len(params)) if i in self.result_idx]
        for i in ordered:
            param = params[i]
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tirx.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j, shape_scale(buffer, j))
        for i in ordered:
            param = params[i]
            if param in buffer_map:
                buffer = buffer_map[param]
                stride_scale = _storage_pack_factor(buffer.dtype)
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
            return _storage_pack_factor(buffer.dtype) if dim_idx == len(buffer.shape) - 1 else 1

        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        sources: dict[str, list[tuple[int, int, int]]] = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                stride_scale = _storage_pack_factor(buffer.dtype)
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
                buffer_dtype_map[name] = (i, _accepted_storage_dtypes(dtype))
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
            pack_factor = _storage_pack_factor(param.dtype)
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
                packing_factor = _storage_pack_factor(buffer.dtype)
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

    def _process_buffer_device(self) -> dict[str, tuple[int, torch.device]]:
        device = torch.device("npu")
        func = self.prim_func
        return {func.buffer_map[param].name: (i, device) for i, param in enumerate(func.params) if param in func.buffer_map}

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Low-level function to call the compiled PTO kernel.

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
                stream: NPU stream ID, default to -1, will use the current stream if not specified
                skip_tensor_validation: Whether to skip tensor attributes validation which
                includes shape, dtype, device, etc.
            """
            return self.runtime_wrapper.forward([*args], stream=stream, skip_tensor_validation=skip_tensor_validation)

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
