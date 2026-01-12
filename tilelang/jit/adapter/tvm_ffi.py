"""Utilities to adapt TVM FFI kernels to Torch tensors.

This adapter intentionally captures PyTorch's current CUDA stream and device
via light-weight callables so that, when the wrapped function is invoked,
the execution observes the same stream context as the active Torch code.
On non-CUDA builds, the stream/device fall back to 0/CPU semantics.
"""

from __future__ import annotations

from typing import Callable, Any
import sys

import torch
from tilelang import tvm
from tvm import runtime, tir
from tvm.target import Target
from tvm.relax import TensorType
from tilelang.utils.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.language import retrieve_func_from_module
from tilelang.engine.param import KernelParam
from tilelang.language.dtypes import dtype


COMPILE_ARGS = {}

if sys.platform == "darwin":
    from torch.utils import cpp_extension

    COMPILE_ARGS["options"] = ["-x", "objective-c++", "-g", "-std=gnu++17"] + ["-I" + i for i in cpp_extension.include_paths()]


class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors.

    Notes
    - We capture the "current" PyTorch CUDA stream/device as thunks (callables)
      rather than materializing them at construction time. This ensures the
      actual stream/device is read just-in-time when the function runs, matching
      the user's current Torch context (e.g., after a stream guard/switch).
    - The stream pointer returned is a raw CUDA stream handle compatible with
      TVM's device API; on CPU or when CUDA is unavailable, we return 0.
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    executable: tvm.runtime.Executable | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None
    # host_mod
    host_mod: tvm.IRModule | None = None
    # device_mod
    device_mod: tvm.IRModule | None = None
    # rt_mod
    rt_mod: tvm.runtime.Module | None = None
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tir.Var, tuple[int, int, int]] | None = None

    # Stream/device functors are inherited from BaseKernelAdapter
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
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

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.kernel_global_source = self.device_kernel_source

        self._post_init()

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
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
            if isinstance(param, tir.Var) and (param not in dynamic_symbolic_map):
                dynamic_symbolic_map[param] = (2, i, -1)
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

    def _convert_torch_func(self) -> Callable[..., Any]:
        # Capture thunks that reflect Torch's current stream and device.
        # These are evaluated at call time to align TVM execution with the
        # caller's active PyTorch stream/device.
        # current_stream_functor = self.get_current_stream_functor()
        current_device_functor = self.get_current_device_functor()

        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        param_dtypes = [param.torch_dtype() for param in self.params]
        # Convert TVM shape arrays to native Python lists
        param_shapes = []

        for param in self.params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            tl_dtype = param.dtype
            if tl_dtype.bits < 8:
                stroage_dtype: dtype = dtype(param.torch_dtype())
                # last dim divide by bits to get the actual shape
                native_shape[-1] = native_shape[-1] * tl_dtype.bits * tl_dtype.lanes // (stroage_dtype.bits * stroage_dtype.lanes)
            param_shapes.append(native_shape)

        if self.executable is None:
            self.executable = runtime.Executable(self.rt_mod)
            if COMPILE_ARGS:
                # Precompile jit module with extra arguments
                self.executable.jit(**COMPILE_ARGS)

        dynamic_symbolic_map = self._process_dynamic_symbolic()
        executable = self.executable

        # Prepare helpers for friendly dtype error messages
        prim_func = self.prim_func
        buffer_map = prim_func.buffer_map
        params = prim_func.params
        # Expected dtype string per parameter index (for buffers only)
        expected_dtype_strs: list[str | None] = []
        # Track whether each param is a buffer (has dtype) vs scalar
        is_buffer_param: list[bool] = []
        for p in params:
            if p in buffer_map:
                expected_dtype_strs.append(str(buffer_map[p].dtype))
                is_buffer_param.append(True)
            else:
                expected_dtype_strs.append(None)
                is_buffer_param.append(False)

        def func(*inputs: torch.Tensor | Any):
            # Validate input count strictly
            expected_inputs = len(self.params) - len(self.result_idx)
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")

            # Resolve the device used for outputs. Prefer the first tensor input's device
            # if available, otherwise use PyTorch's current device.
            out_device: torch.device | None = None

            # Stitch the full positional argument list expected by the TVM executable
            ins_idx: int = 0
            tensor_list: list[torch.Tensor] = []

            # Prepare input and output tensors
            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = param_dtypes[i]
                    shape = []
                    # Now working with native Python list, no FFI calls needed
                    for s in param_shapes[i]:
                        if isinstance(s, tir.Var):
                            for key in dynamic_symbolic_map:
                                if str(s) == str(key):
                                    ref_id, ref_tensor_idx, ref_shape_idx = dynamic_symbolic_map[key]
                                    if ref_id == 2:
                                        shape.append(inputs[ref_tensor_idx])
                                    elif ref_id == 0:
                                        shape.append(tensor_list[ref_tensor_idx].shape[ref_shape_idx])
                                    elif ref_id == 1:
                                        shape.append(tensor_list[ref_tensor_idx].stride()[ref_shape_idx])
                        else:  # Already converted to Python int during initialization
                            shape.append(s)

                    if out_device is None:
                        out_device = current_device_functor()

                    if len(shape) == 0:
                        param_name = self.params[i].name if hasattr(self.params[i], "name") else f"parameter_{i}"
                        raise ValueError(
                            f"Cannot create output tensor (name={param_name}) - 0-dimensional tensors are not supported. "
                            f"Expected shape: {shape}"
                        )
                    tensor = torch.empty(*shape, dtype=dtype, device=out_device)
                else:
                    tensor = inputs[ins_idx]
                    ins_idx += 1
                tensor_list.append(tensor)

            executable(*tensor_list)

            # Return outputs in the requested form
            if len(self.result_idx) == 1:
                return tensor_list[self.result_idx[0]]
            return [tensor_list[i] for i in self.result_idx]

        return func

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
        adapter.wrapped_source = device_kernel_source + "\n\n" + host_kernel_source
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target.canon_target(determine_target(target))

        adapter.verbose = verbose
        adapter.executable = runtime.load_module(kernel_lib_path)
        adapter._post_init()
        return adapter

    def get_host_source(self):
        """Returns the source code of the host module."""
        if self.host_kernel_source is not None:
            return self.host_kernel_source
        return self.rt_mod.inspect_source()

    def get_device_source(self):
        """Returns the source code of the device module."""
        if self.device_kernel_source is not None:
            return self.device_kernel_source
        return self.rt_mod.imports[0].inspect_source()

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.get_device_source()
        else:
            return self.get_device_source() + "\n\n" + self.get_host_source()

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
