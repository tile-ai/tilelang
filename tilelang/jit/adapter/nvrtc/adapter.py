from __future__ import annotations
import logging
from typing import Any, Callable

import torch
from tvm import tir
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter
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
        func_or_mod: tir.PrimFunc | tvm.IRModule,
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

        if isinstance(func_or_mod, tir.PrimFunc):
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
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    # Keep tir.Var for dynamic dimensions
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        self.target = Target.canon_target(determine_target(target))
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

        if isinstance(func_or_mod, tir.PrimFunc):
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
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    # Keep tir.Var for dynamic dimensions
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            adapter.param_shapes.append(native_shape)

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()

        adapter.target = Target.canon_target(determine_target(target))
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

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.

        Returns
        -------
        Dict[tir.Var, Tuple[int, int]]
            Mapping from symbolic variable to (buffer_index, shape_dimension)
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                    dynamic_symbolic_map[shape] = (i, j)
        return dynamic_symbolic_map

    def get_kernel_source(self, kernel_only: bool = True) -> str | None:
        """Get the CUDA kernel source code.

        Returns
        -------
        Optional[str]
            The kernel source code, or None if not available
        """
        if kernel_only:
            return self.device_kernel_source
        else:
            return self.host_func

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Low-level function to call the compiled CUDA kernel."""
        return self.pymodule.call(self.kernels, *args, stream=stream)

    def _wrap_forward_from_prebuild_lib(self, *ins: list[torch.Tensor], stream: int | None = None):
        """High-level wrapper for kernel execution.

        Handles:
        1. Input validation
        2. Output tensor allocation
        3. Dynamic shape resolution
        4. CUDA stream management

        Args:
            ins: Input PyTorch tensors
            stream: Optional CUDA stream for asynchronous execution

        Returns:
            Single tensor or list of tensors containing the kernel results
        """
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )
        ins_idx = 0
        args = []

        # tensor pointers
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tir.Var):
                        ref_tensor_idx, ref_shape_idx = self.dynamic_symbolic_map[s]
                        shape.append(ins[ref_tensor_idx].shape[ref_shape_idx])
                    else:  # Already converted to Python int during initialization
                        shape.append(s)
                device = ins[0].device if len(ins) > 0 else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                tensor = ins[ins_idx]
                ins_idx += 1
            args.append(tensor)

        # dynamic symbolics
        for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
            args.append(ins[buffer_idx].shape[shape_idx])

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
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
