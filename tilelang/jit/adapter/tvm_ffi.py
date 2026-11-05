"""Utilities to adapt TVM-FFI kernels to Torch tensors.

TVM-FFI obtains the active work stream through Torch's DLPack Exchange API.
The Ascend adapter installs TileLang's Torch NPU callback before the first
tensor reaches an executable.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable
import sys
import threading

import torch
import tvm_tirx
import tvm_ffi
from tvm_tirx import runtime, tirx
from tvm_tirx.codegen import Executable
from tvm_tirx.target import Target
from tilelang.backend.target import determine_target
from tilelang.jit.abi import CALLEE_ALLOCATED_OUTPUTS_ATTR, FunctionABIPlan
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.engine.param import KernelParam


COMPILE_ARGS = {}

if sys.platform == "darwin":
    from torch.utils import cpp_extension

    COMPILE_ARGS["options"] = ["-x", "objective-c++", "-g", "-std=gnu++17"] + ["-I" + i for i in cpp_extension.include_paths()]
elif sys.platform == "win32":
    from tilelang.contrib.msvc import create_shared as _msvc_create_shared

    COMPILE_ARGS["fcompile"] = _msvc_create_shared


def _install_torch_stream_exchange() -> None:
    from tilelang.ascend.torch_exchange import (
        install_torch_npu_stream_exchange,
    )

    install_torch_npu_stream_exchange()


class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors.

    Notes
    - Torch tensors use TVM-FFI's zero-copy DLPack Exchange API conversion.
    - Ascend execution installs a Cython callback that reads Torch's current
      NPU stream for every invocation.
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm_tirx.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    executable: Executable | tvm_ffi.Module | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None
    # host_mod
    host_mod: tvm_tirx.IRModule | None = None
    # device_mod
    device_mod: tvm_tirx.IRModule | None = None
    # rt_mod
    rt_mod: tvm_ffi.Module | None = None
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tirx.Var, tuple[int, int, int, int]] | None = None

    _torch_npu_stream_exchange_installed: bool = False

    def _prepare_torch_device(self, device: torch.device) -> None:
        if device.type == "npu" and not self._torch_npu_stream_exchange_installed:
            _install_torch_stream_exchange()
            self._torch_npu_stream_exchange_installed = True

    # Stream/device functors are inherited from BaseKernelAdapter
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.Function | tvm_tirx.IRModule,
        host_mod: tvm_tirx.IRModule | None = None,
        device_mod: tvm_tirx.IRModule | None = None,
        rt_mod: tvm_ffi.Module | None = None,
        host_kernel_source: str | None = None,
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
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tirx.Function):
            self.ir_module = tvm_tirx.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target(determine_target(target))

        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self._ffi_callee_allocated_output_abi = self._uses_ffi_callee_allocated_output_abi()
        self._abi_plan = FunctionABIPlan(self.prim_func, self.result_idx, tensor_error_type=RuntimeError)
        self.dynamic_symbolic_map = None if self._ffi_callee_allocated_output_abi else self._process_dynamic_symbolic()
        self.kernel_global_source = self.device_kernel_source
        self.executable = None
        self._executable_lock = threading.Lock()

        self._post_init()

    def _make_executable(self) -> Executable:
        if self.rt_mod is None:
            raise RuntimeError("Cannot create TVM FFI executable without a runtime module.")
        executable = Executable(self.rt_mod)
        compile_args = self.get_export_kwargs()
        if compile_args:
            executable.jit(**compile_args)
        return executable

    def get_export_kwargs(self) -> dict:
        """Use the same source and link options for JIT, disk cache and export."""
        compile_args = dict(COMPILE_ARGS)
        from tilelang.cpu.toolchain import get_compile_flags

        backend_flags = get_compile_flags(self.target, self.pass_configs, execution_backend="tvm_ffi")
        if self.target.kind.name == "c":
            from tilelang.env import TILELANG_TEMPLATE_PATH

            # C source is compiled at export/JIT time. Device compiler options
            # belong to their respective backend paths; reinterpreting them
            # as host linker options changes their meaning.
            backend_flags += list(self.compile_flags or ())
            backend_flags += ["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH]
        if backend_flags:
            options = list(compile_args.get("options", []))
            options += backend_flags
            compile_args["options"] = options
        return compile_args

    def _get_executable(self) -> Executable | tvm_ffi.Module:
        executable = self.executable
        if executable is not None:
            return executable

        with self._executable_lock:
            executable = self.executable
            if executable is None:
                executable = self._make_executable()
                self.executable = executable
            return executable

    def get_exportable_executable(self) -> Executable | tvm_ffi.Module:
        """Return the lazy executable, or the runnable module loaded from disk cache."""
        return self._get_executable()

    def _uses_ffi_callee_allocated_output_abi(self) -> bool:
        """Whether lowering gives this kernel the callee-allocated main ABI."""
        if not self.result_idx:
            return False

        # The ABI decision is stamped on the PrimFunc at compile time from the
        # execution backend's declared capability; read it back rather than
        # re-deriving it from the target, so lowering and dispatch can never
        # disagree.
        attrs = getattr(self.prim_func, "attrs", None)
        if attrs is None or CALLEE_ALLOCATED_OUTPUTS_ATTR not in attrs:
            return False
        return int(attrs[CALLEE_ALLOCATED_OUTPUTS_ATTR]) != 0

    def _process_dynamic_symbolic(self) -> dict:
        """Expose direct native shape sources for adapter diagnostics.

        Allocation uses FunctionABIPlan, which also resolves linear expressions
        and validates repeated constraints. This view retains the established
        field/parameter/dimension/scale encoding for direct symbols only.
        """
        from tvm_tirx import ir

        sources = {}
        for index in self._abi_plan.input_indices:
            parameter = self.prim_func.params[index]
            if isinstance(parameter.ty, ir.PrimType):
                sources[parameter] = (2, index, -1, 1)
        for index in self._abi_plan.input_indices:
            tensor = self._abi_plan.tensors.get(index)
            if tensor is None:
                continue
            for axis, expression in enumerate(tensor.shape):
                if isinstance(expression, ir.Var) and expression not in sources:
                    sources[expression] = (0, index, axis, 1)
            bits = tensor.dtype.bits * tensor.dtype.lanes
            for axis, expression in enumerate(tensor.strides):
                if isinstance(expression, ir.Var) and expression not in sources:
                    scale = 8 // bits if bits < 8 and axis + 1 < len(tensor.strides) else 1
                    sources[expression] = (1, index, axis, scale)
        return sources

    def _get_output_device_functor(self) -> Callable[[], torch.device]:
        # A CPU executable stays on CPU when there is no tensor input to carry
        # its device. Runtime availability must not redirect scalar-only or
        # nullable-input kernels to an unrelated GPU allocator.
        if Target(self.target).get_target_device_type() == 1:  # DLPack kDLCPU
            return lambda: torch.device("cpu")
        return self.get_current_device_functor()

    def _convert_torch_func(self) -> Callable[..., Any]:
        if getattr(self, "_ffi_callee_allocated_output_abi", False):
            return self._convert_ffi_callee_allocated_output_func()

        current_device_functor = None
        output_dtypes = {index: self.params[index].torch_dtype() for index in self.result_idx}
        output_storage_bits = {index: torch.empty((), dtype=dtype).element_size() * 8 for index, dtype in output_dtypes.items()}

        def func(*inputs: torch.Tensor | Any):
            nonlocal current_device_functor
            if self.result_idx:
                tensor_list, bindings = self._abi_plan.bind_inputs(inputs)
            else:
                # No output allocation needs Python shape bindings. Let the
                # native entry validate scalar kinds and tensor descriptors,
                # preserving its public argument-error contract.
                expected_inputs = len(self.params)
                if len(inputs) != expected_inputs:
                    raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")
                tensor_list = list(inputs)
            out_device = next((value.device for value in inputs if isinstance(value, torch.Tensor)), None)
            for index in self.result_idx:
                dtype = output_dtypes[index]
                storage_bits = output_storage_bits[index]
                shape = self._abi_plan.output_shape(index, bindings, storage_bits=storage_bits)
                if not shape:
                    raise ValueError(
                        f"Cannot create output tensor (parameter={index}) - 0-dimensional tensors are not supported. "
                        f"Expected shape: {shape}"
                    )
                if out_device is None:
                    if current_device_functor is None:
                        current_device_functor = self._get_output_device_functor()
                    out_device = current_device_functor()
                tensor_list[index] = torch.empty(shape, dtype=dtype, device=out_device)
            if out_device is not None:
                self._prepare_torch_device(out_device)
            self._get_executable()(*tensor_list)
            if len(self.result_idx) == 1:
                return tensor_list[self.result_idx[0]]
            return [tensor_list[index] for index in self.result_idx]

        return func

    def _convert_ffi_callee_allocated_output_func(self) -> Callable[..., Any]:
        """Create a Torch callable whose outputs are allocated by TVM-FFI."""
        current_device_functor = None
        expected_inputs = len(self.params) - len(self.result_idx)

        def func(*inputs: torch.Tensor | Any):
            nonlocal current_device_functor
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")

            allocator_anchor = next((value for value in inputs if isinstance(value, torch.Tensor)), None)
            if allocator_anchor is None:
                if current_device_functor is None:
                    current_device_functor = self._get_output_device_functor()
                device = current_device_functor()
            else:
                device = allocator_anchor.device

            self._prepare_torch_device(device)
            if not (hasattr(torch.Tensor, "__dlpack_c_exchange_api__") or hasattr(torch.Tensor, "__c_dlpack_exchange_api__")):
                raise RuntimeError(
                    "TVM-FFI callee-allocated outputs require Torch's DLPack allocator exchange API. "
                    "Install a compatible torch-c-dlpack-ext or use a supported PyTorch build."
                )

            if allocator_anchor is None:
                # A Torch tensor argument installs the EnvTensorAllocator in
                # TVM-FFI's thread-local call context.  Scalar-only kernels use
                # a zero-element anchor solely for that allocator/device state.
                allocator_anchor = torch.empty(0, device=device)

            result = self._get_executable()(*inputs, allocator_anchor)
            if len(self.result_idx) == 1:
                return result
            return list(result)

        return func

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        target: str,
        func_or_mod: tirx.Function | tvm_tirx.IRModule,
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
        host_kernel_source = adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        device_kernel_source = adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        adapter.wrapped_source = (
            device_kernel_source.text + "\n\n" + host_kernel_source.text
            if device_kernel_source.text is not None and host_kernel_source.text is not None
            else None
        )
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tirx.Function):
            adapter.ir_module = tvm_tirx.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target(determine_target(target))

        adapter.verbose = verbose
        adapter.libpath = kernel_lib_path
        adapter.kernel_global_source = device_kernel_source.text
        adapter.rt_mod = None
        adapter.executable = runtime.load_module(kernel_lib_path)
        adapter._ffi_callee_allocated_output_abi = adapter._uses_ffi_callee_allocated_output_abi()
        adapter._abi_plan = FunctionABIPlan(adapter.prim_func, adapter.result_idx, tensor_error_type=RuntimeError)
        adapter.dynamic_symbolic_map = None if adapter._ffi_callee_allocated_output_abi else adapter._process_dynamic_symbolic()
        adapter._executable_lock = threading.Lock()
        adapter._post_init()
        return adapter

    def get_host_source(self) -> str | None:
        """Returns the source code of the host module."""
        source = self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
        if source is not None:
            return source
        rt_mod = getattr(self, "rt_mod", None)
        if rt_mod is None:
            return None
        return rt_mod.inspect_source()

    def get_device_source(self) -> str | None:
        """Returns the source code of the device module."""
        source = self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        if source is not None:
            self.kernel_global_source = source
            return source
        rt_mod = getattr(self, "rt_mod", None)
        if rt_mod is None:
            return None
        if rt_mod.kind == "tilelang.output":
            rt_mod = rt_mod.imports[0]
        return rt_mod.imports[0].inspect_source()

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        device_source = self.get_device_source() or ""
        if kernel_only:
            return device_source

        host_source = self.get_host_source() or ""
        if device_source and host_source:
            return device_source + "\n\n" + host_source
        return device_source or host_source

    @property
    def prim_func(self) -> tirx.Function:
        """Return the native source Function, independent of frontend imports."""
        functions = list(self.ir_module.functions.values())
        if len(functions) != 1 or not isinstance(functions[0], tirx.Function):
            raise ValueError("A kernel adapter requires exactly one native Function")
        return functions[0]
