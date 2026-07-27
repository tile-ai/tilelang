from __future__ import annotations
import logging
from typing import Any
from collections.abc import Callable

import torch
from tvm import tirx
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.language import retrieve_func_from_module
from tilelang.backend.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.jit.adapter.nvrtc import is_nvrtc_available, check_nvrtc_available

from .libgen import NVRTCLibraryGenerator

logger = logging.getLogger(__name__)

# Import cuda bindings if available
if is_nvrtc_available:
    import cuda.bindings.driver as cuda


class NVRTCKernelAdapter(BaseKernelAdapter):
    pymodule = None
    kernels = {}

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
        check_nvrtc_available()

        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tirx.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        # Cache parameter information during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        self.param_dtypes = [param.torch_dtype() for param in params]
        self.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    # Keep tirx.Var for dynamic dimensions
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        self.target = Target(determine_target(target))
        self.verbose = verbose
        self.wrapper = TLPyWrapper(self.target)
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        wrapper_result = self.wrapper.wrap(device_kernel_source)
        self.host_func = wrapper_result["host_func"]
        self.function_names = wrapper_result["function_names"]

        self.lib_generator = NVRTCLibraryGenerator(self.target, self.verbose)
        self.lib_generator.update_lib_code(self.device_kernel_source)
        self.lib_generator.update_host_func(self.host_func)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.compile_lib()
        self.lib_generator.load_lib()
        self.libpath = self.lib_generator.libpath
        self.pymodule = self.lib_generator.pymodule
        culib = self.lib_generator.culib
        for name in self.function_names:
            result, self.kernels[name] = cuda.cuLibraryGetKernel(culib, bytes(name, "utf-8"))
            assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to get kernel: {name}"

        self._post_init()

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
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
        host_kernel_source = adapter._set_cached_text_source("host_func", "_host_kernel_source_path", host_kernel_source)
        adapter.host_kernel_source = host_kernel_source.text
        adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)

        if isinstance(func_or_mod, tirx.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        # Cache parameter information during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        adapter.param_dtypes = [param.torch_dtype() for param in params]
        adapter.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    # Keep tirx.Var for dynamic dimensions
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            adapter.param_shapes.append(native_shape)

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()

        adapter.target = Target(determine_target(target))
        adapter.verbose = verbose
        adapter.lib_generator = NVRTCLibraryGenerator(adapter.target, adapter.verbose)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.pymodule = adapter.lib_generator.pymodule
        adapter.function_names = adapter.pymodule._function_names

        culib = adapter.lib_generator.culib
        for name in adapter.function_names:
            result, adapter.kernels[name] = cuda.cuLibraryGetKernel(culib, bytes(name, "utf-8"))
            assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to get kernel: {name}"

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> dict[tirx.Var, tuple[int, int, int, int]]:
        """Extract runtime sources for scalar, shape, and stride symbols.

        Each entry contains ``(kind, parameter_index, dimension, scale)``.
        ``kind`` is 0 for a shape, 1 for a stride, and 2 for an explicit
        scalar parameter. Shape symbols precede stride symbols to match the
        generated NVRTC host wrapper ABI.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map: dict[tirx.Var, tuple[int, int, int, int]] = {}
        self._dynamic_symbolic_candidates_map: dict[tirx.Var, list[tuple[int, int, int, int]]] = {}
        self._dynamic_symbolic_name_candidates_map: dict[str, list[tuple[int, int, int, int]]] = {}
        # Secondary index by variable name for fallback lookup when tirx.Var
        # object identity differs (e.g. params created from a different
        # PrimFunc instance than the one stored in ir_module).
        self._dynamic_symbolic_name_map: dict[str, tuple[int, int, int, int]] = {}

        def unique_push_back(v: tirx.Var, entry: tuple[int, int, int, int]):
            self._dynamic_symbolic_candidates_map.setdefault(v, []).append(entry)
            self._dynamic_symbolic_name_candidates_map.setdefault(v.name, []).append(entry)
            if v in dynamic_symbolic_map or v.name in self._dynamic_symbolic_name_map:
                return
            dynamic_symbolic_map[v] = entry
            self._dynamic_symbolic_name_map[v.name] = entry

        # Explicit scalar parameters are already part of the wrapper's primary
        # argument list. Recording them here permits output-shape resolution,
        # but they must not be appended again as implicit dynamic arguments.
        for i, param in enumerate(params):
            if param not in buffer_map:
                unique_push_back(param, (2, i, -1, 1))

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tirx.Var):
                    unique_push_back(shape, (0, i, j, 1))

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            element_bits = buffer.dtype.bits * buffer.dtype.lanes
            stride_scale = 8 // element_bits if element_bits < 8 else 1
            for j, stride in enumerate(buffer.strides):
                if isinstance(stride, tirx.Var):
                    unique_push_back(stride, (1, i, j, stride_scale))

        return dynamic_symbolic_map

    def _lookup_dynamic_symbolic_candidates(self, v: tirx.Var) -> list[tuple[int, int, int, int]]:
        """Return all scalar, shape, and stride sources for a dynamic symbol."""
        if v in self._dynamic_symbolic_candidates_map:
            return self._dynamic_symbolic_candidates_map[v]
        if v.name in self._dynamic_symbolic_name_candidates_map:
            return self._dynamic_symbolic_name_candidates_map[v.name]
        raise KeyError(f"Dynamic symbolic variable '{v.name}' not found in symbolic map")

    def _resolve_dynamic_symbolic_value(self, v: tirx.Var, param_values: list[Any]):
        """Resolve one symbolic value from the first available runtime source."""
        unavailable_sources = []
        for ref_id, param_idx, dim_idx, stride_scale in self._lookup_dynamic_symbolic_candidates(v):
            ref_value = param_values[param_idx]
            if ref_id == 2 and ref_value is not None:
                return ref_value
            if isinstance(ref_value, torch.Tensor):
                if ref_id == 0:
                    return ref_value.shape[dim_idx]
                if ref_id == 1:
                    return ref_value.stride()[dim_idx] * stride_scale
                raise ValueError(f"Unknown dynamic symbolic reference kind: {ref_id}")
            unavailable_sources.append(f"parameter {param_idx}: {type(ref_value).__name__}")
        details = ", ".join(unavailable_sources)
        raise TypeError(f"Dynamic symbolic variable '{v.name}' has no available runtime source ({details})")

    def get_kernel_source(self, kernel_only: bool = True) -> str | None:
        """Get the CUDA kernel source code.

        Returns
        -------
        Optional[str]
            The kernel source code, or None if not available
        """
        if kernel_only:
            return self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        else:
            return self._load_cached_text_source("host_func", "_host_kernel_source_path")

    def get_host_source(self) -> str | None:
        """Get the cached host-side source code."""
        return self._load_cached_text_source("host_func", "_host_kernel_source_path")

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Low-level function to call the compiled CUDA kernel."""
        return self.pymodule.call(self.kernels, *args, stream=stream)

    def _wrap_forward_from_prebuild_lib(self, *ins: Any, stream: int | None = None):
        """High-level wrapper for kernel execution.

        Handles:
        1. Input validation
        2. Output tensor allocation
        3. Dynamic shape resolution
        4. CUDA stream management

        Args:
            ins: Input PyTorch tensors and scalar values
            stream: Optional CUDA stream for asynchronous execution

        Returns:
            Single tensor or list of tensors containing the kernel results
        """
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )
        ins_idx = 0
        param_values: list[Any] = [None] * len(self.params)
        for i in range(len(self.params)):
            if i in self.result_idx:
                continue
            param_values[i] = ins[ins_idx]
            ins_idx += 1

        first_tensor = next((value for value in param_values if isinstance(value, torch.Tensor)), None)

        # Allocate output tensors in their PrimFunc parameter positions.
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tirx.Var):
                        shape.append(self._resolve_dynamic_symbolic_value(s, param_values))
                    else:  # Already converted to Python int during initialization
                        shape.append(s)
                device = first_tensor.device if first_tensor is not None else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
                param_values[i] = tensor

        # dynamic symbolics
        args = list(param_values)
        for symbol, (ref_id, _, _, _) in self.dynamic_symbolic_map.items():
            if ref_id != 2:
                args.append(self._resolve_dynamic_symbolic_value(symbol, param_values))

        # if stream is not None, we need to pass the stream to the library
        if stream is None:
            if str(self.target).startswith("cuda") and torch.cuda.is_available():
                stream = torch.cuda.current_stream().cuda_stream
            else:
                stream = 0

        self._forward_from_prebuild_lib(*args, stream=stream)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        else:
            return [args[i] for i in self.result_idx]

    def _convert_torch_func(self) -> Callable[..., torch.Tensor | list[torch.Tensor]]:
        """Convert to a PyTorch-compatible function.

        Returns
        -------
        Callable[..., Union[torch.Tensor, List[torch.Tensor]]]
            A callable function that takes tensors and returns tensor(s)
        """
        return self._wrap_forward_from_prebuild_lib

    @property
    def prim_func(self) -> tirx.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
