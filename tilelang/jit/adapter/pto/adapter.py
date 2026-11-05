"""Compile and execute PTODSL kernels using the PTO launch wrapper."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import torch
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from tilelang_pto_wrapper import PTOKernelWrapper

from tilelang.backend.target import determine_target
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.utils.language import retrieve_func_from_module
from tilelang.jit.adapter.abi import source_abi_plan

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
    from tilelang.language import dtype as language_dtype

    torch_dtype = language_dtype(dtype).as_torch()
    if _storage_pack_factor(dtype) > 1 and torch_dtype != torch.int8:
        return (torch_dtype, torch.int8)
    return torch_dtype


def _validate_packed_layout(plan):
    """PTO stores adjacent sub-byte values in one byte, including outputs."""
    for tensor in plan.tensors.values():
        factor = _storage_pack_factor(tensor.dtype)
        if factor == 1 or not tensor.shape:
            continue
        extent = tensor.shape[-1]
        if isinstance(extent, ir.IntImm) and int(extent) % factor:
            raise ValueError(f"The innermost dimension of {tensor.dtype} must be divisible by its packing factor ({factor})")
        stride = tensor.strides[-1]
        if not isinstance(stride, ir.IntImm) or int(stride) != 1:
            raise ValueError(f"Packed {tensor.dtype} buffers require a static innermost stride of 1")
        for stride in tensor.strides[:-1]:
            if isinstance(stride, ir.IntImm) and int(stride) % factor:
                raise ValueError(f"The stride of packed {tensor.dtype} must be divisible by its packing factor ({factor})")


class PTOKernelAdapter(BaseKernelAdapter):
    """Own PTO compilation and cached-library loading with a Cython launch ABI."""

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int | None,
        target: str | dict[str, object] | Target,
        func_or_mod: tirx.Function | tvm.IRModule,
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
        func_or_mod: tirx.Function | tvm.IRModule,
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
        if isinstance(func_or_mod, tirx.Function):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
        self.target = Target(determine_target(target))
        self.verbose = verbose
        self.pass_configs = pass_configs

        self.source_abi = source_abi_plan(self)
        self.params = self.source_abi.validate_parameter_metadata(self.params)
        _validate_packed_layout(self.source_abi)
        self.buffer_dtype_map = {
            tensor.variable: (index, _accepted_storage_dtypes(tensor.dtype)) for index, tensor in self.source_abi.tensors.items()
        }
        self.buffer_device_map = {tensor.variable: (index, torch.device("npu")) for index, tensor in self.source_abi.tensors.items()}
        self.ptr_map = {
            index: parameter.name
            for index, parameter in enumerate(self.prim_func.params)
            if isinstance(parameter.ty, (ir.PtrType, tirx.TensorType))
        }

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
        self.runtime_wrapper.set_source_abi_plan(self.source_abi)
        self.runtime_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.runtime_wrapper.set_buffer_device_map(self.buffer_device_map)
        self.runtime_wrapper.set_ptr_map(self.ptr_map)
        self._post_init()

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
        return bool(self.source_abi.implicit_symbols)

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
