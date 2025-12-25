from __future__ import annotations
import logging
import weakref
from typing import Any, Callable

import torch
from tvm import tir
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available
from tilelang.jit.adapter.cutedsl.libgen import CuTeDSLLibraryGenerator
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter

logger = logging.getLogger(__name__)


class CuTeDSLKernelAdapter(BaseKernelAdapter):
    pymodule = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        check_cutedsl_available()

        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tir.PrimFunc):
            gsym = func_or_mod.attrs.get("global_symbol")
            if gsym is None:
                raise ValueError("PrimFunc is missing required attr 'global_symbol'")
            self.ir_module = tvm.IRModule({gsym: func_or_mod})
        else:
            self.ir_module = func_or_mod

        # Cache parameter information during initialization
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

        self.dynamic_symbolic_map, self.dynamic_symbolic_order = self._process_dynamic_symbolic()

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
        self.tma_cpp_init_code = wrapper_result["tma_cpp_init_code"]
        self.tma_lib_name = wrapper_result["tma_lib_name"]
        self.launcher_cpp_code = wrapper_result.get("launcher_cpp_code", None)
        self.launcher_lib_name = wrapper_result.get("launcher_lib_name", None)

        self.lib_generator = CuTeDSLLibraryGenerator(self.target, self.verbose)
        self.lib_generator.update_lib_code(self.device_kernel_source)
        self.lib_generator.update_host_func(self.host_func)
        self.lib_generator.update_tma_cpp_init_code(self.tma_cpp_init_code)
        self.lib_generator.update_tma_lib_name(self.tma_lib_name)
        self.lib_generator.update_launcher_cpp_code(self.launcher_cpp_code)
        self.lib_generator.update_launcher_lib_name(self.launcher_lib_name)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.compile_lib()
        self.lib_generator.load_lib()
        self.libpath = self.lib_generator.libpath
        self.device_kernel_source = open(self.libpath).read()
        self.pymodule = self.lib_generator.pymodule

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
            gsym = func_or_mod.attrs.get("global_symbol")
            if gsym is None:
                raise ValueError("PrimFunc is missing required attr 'global_symbol'")
            adapter.ir_module = tvm.IRModule({gsym: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        # Cache parameter information during initialization
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

        adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()

        adapter.target = Target.canon_target(determine_target(target))
        adapter.verbose = verbose
        adapter.lib_generator = CuTeDSLLibraryGenerator(adapter.target, adapter.verbose)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.libpath = kernel_lib_path
        adapter.kernel_global_source = open(adapter.libpath).read()
        adapter.pymodule = adapter.lib_generator.pymodule

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> tuple[dict[tir.Var, tuple[int, int, int]], list[tir.Var]]:
        """Extract information about dynamic symbols from the TIR function.

        We follow the same ordering semantics as `TLCUDASourceWrapper.get_dynamic_symbolic_set()`:
        1) dynamic symbols in buffer shapes (in prim_func param order)
        2) then dynamic symbols in buffer strides

        The mapping encodes:
        - id=0: shape var -> (0, buffer_param_index, dim_index)
        - id=1: stride var -> (1, buffer_param_index, stride_index)

        Returns:
            (dynamic_symbolic_map, dynamic_symbolic_order)
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map: dict[tir.Var, tuple[int, int, int]] = {}
        dynamic_symbolic_order: list[tir.Var] = []

        def unique_push_back(v: tir.Var, entry: tuple[int, int, int]):
            if v in dynamic_symbolic_map:
                return
            dynamic_symbolic_map[v] = entry
            dynamic_symbolic_order.append(v)

        # 1) Shapes
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var):
                    unique_push_back(shape, (0, i, j))

        # 2) Strides
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            if buffer.strides is None:
                continue
            for j, stride in enumerate(buffer.strides):
                if isinstance(stride, tir.Var):
                    unique_push_back(stride, (1, i, j))

        return dynamic_symbolic_map, dynamic_symbolic_order

    def get_kernel_source(self, kernel_only: bool = True) -> str | None:
        """Get the CUDA kernel source code.

        Returns
        -------
        str | None
            The kernel source code, or None if not available
        """
        return self.device_kernel_source

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None, device_id: int = 0):
        """Low-level function to call the compiled CUDA kernel.

        Args:
            *args: Kernel arguments (tensors and scalars)
            stream: CUDA stream handle
            device_id: CUDA device ID for multi-GPU support
        """
        result = self.pymodule.call(*args, stream=stream, device_id=device_id)

        # After first call, save cubin to cache if needed
        self._save_cubin_to_cache_if_needed()

        return result

    def _save_cubin_to_cache_if_needed(self):
        """Save cubin to cache directory after first execution.

        This is called after the first kernel execution to ensure the generated
        cubin file is copied to the cache directory for future reuse.
        """
        if getattr(self, "_cubin_saved_to_cache", False):
            return
        self._cubin_saved_to_cache = True

        # Check if we have a cache path (set by kernel_cache)
        cache_path = getattr(self, "_cache_path", None)
        if cache_path is None:
            return

        import os
        import shutil

        # Source cubin path (in temp directory)
        src_py_path = self.libpath
        src_py_stem = os.path.splitext(os.path.basename(src_py_path))[0]
        src_dir = os.path.dirname(src_py_path)
        src_cubin_path = os.path.join(src_dir, f"{src_py_stem}.cubin")

        if not os.path.exists(src_cubin_path):
            return

        # Destination cubin path (in cache directory)
        dst_cubin_path = os.path.join(cache_path, "kernel.cubin")

        if os.path.exists(dst_cubin_path):
            return

        # Copy cubin to cache
        try:
            shutil.copy2(src_cubin_path, dst_cubin_path)
            logger.debug(f"Saved CuTeDSL cubin to cache: {dst_cubin_path}")
        except Exception as e:
            logger.warning(f"Failed to save cubin to cache: {e}", exc_info=True)

    def _wrap_forward_from_prebuild_lib(self, *ins: Any, stream: int | None = None):
        """High-level wrapper for kernel execution.

        Handles:
        1. Input validation
        2. Output tensor allocation
        3. Dynamic shape resolution
        4. CUDA stream management

        Args:
            ins: Input arguments (may include scalars and tensors)
            stream: Optional CUDA stream for asynchronous execution

        Returns:
            Single tensor or list of tensors containing the kernel results
        """
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )

        # Materialize args in PrimFunc param order (inputs + allocated outputs)
        ins_idx = 0
        param_values: list[Any] = [None] * len(self.params)
        for i in range(len(self.params)):
            if i in self.result_idx:
                continue
            param_values[i] = ins[ins_idx]
            ins_idx += 1

        first_tensor = next((v for v in param_values if isinstance(v, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Expected at least one torch.Tensor argument to infer CUDA device")

        args: list[Any] = []

        # tensor pointers
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tir.Var):
                        ref_id, ref_param_idx, ref_dim_idx = self.dynamic_symbolic_map[s]
                        ref_val = param_values[ref_param_idx]
                        if not isinstance(ref_val, torch.Tensor):
                            raise TypeError(f"Dynamic shape/stride var {s} refers to a non-tensor param at index {ref_param_idx}")
                        if ref_id == 0:
                            shape.append(ref_val.shape[ref_dim_idx])
                        elif ref_id == 1:
                            # Stride vars are not expected in output shapes, but handle defensively.
                            shape.append(ref_val.stride()[ref_dim_idx])
                        else:
                            raise ValueError(f"Unknown dynamic symbol ref id: {ref_id}")
                    else:  # Already converted to Python int during initialization
                        shape.append(s)
                tensor = torch.empty(*shape, dtype=dtype, device=first_tensor.device)
                param_values[i] = tensor
            else:
                tensor = param_values[i]
            args.append(tensor)

        # dynamic symbolics
        for sym in self.dynamic_symbolic_order:
            ref_id, buffer_idx, dim_idx = self.dynamic_symbolic_map[sym]
            ref_val = param_values[buffer_idx]
            if not isinstance(ref_val, torch.Tensor):
                raise TypeError(f"Dynamic symbolic var {sym} refers to a non-tensor param at index {buffer_idx}")
            if ref_id == 0:
                args.append(ref_val.shape[dim_idx])
            elif ref_id == 1:
                args.append(ref_val.stride()[dim_idx])
            else:
                raise ValueError(f"Unknown dynamic symbol ref id: {ref_id}")

        # if stream is not None, we need to pass the stream to the library
        if stream is None:
            if str(self.target).startswith("cuda") and torch.cuda.is_available():
                stream = torch.cuda.current_stream().cuda_stream
            else:
                stream = 0

        # Get device_id from first tensor for multi-GPU support
        if not first_tensor.is_cuda:
            raise ValueError(f"CuTeDSL kernels require CUDA tensors, got tensor on device: {first_tensor.device}")
        device_id = first_tensor.device.index or 0

        self._forward_from_prebuild_lib(*args, stream=stream, device_id=device_id)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        else:
            return [args[i] for i in self.result_idx]

    def _convert_torch_func(self) -> Callable[..., torch.Tensor | list[torch.Tensor]]:
        """Convert to a PyTorch-compatible function.

        Returns
        -------
        Callable[..., torch.Tensor | list[torch.Tensor]]
            A callable function that takes tensors and returns tensor(s)
        """
        return self._wrap_forward_from_prebuild_lib

    def _post_init(self):
        """Override base class _post_init to register cleanup via weakref.finalize."""
        super()._post_init()

        # Register cleanup for this instance using weakref.finalize
        # This will automatically call cleanup when the object is garbage collected
        if self.pymodule is not None and hasattr(self.pymodule, "cleanup_module"):
            weakref.finalize(self, self._cleanup_module, self.pymodule)

    @staticmethod
    def _cleanup_module(pymodule):
        """Cleanup a single adapter instance's CUDA module and contexts.

        This is called automatically when the adapter instance is garbage collected.
        It can also be called explicitly via the cleanup() instance method.
        """
        try:
            if hasattr(pymodule, "cleanup_module"):
                pymodule.cleanup_module()
        except Exception:
            # Suppress errors during cleanup (might be called during shutdown)
            pass

    def cleanup(self):
        """Explicitly cleanup this adapter's CUDA resources.

        This method can be called explicitly to immediately release CUDA resources
        without waiting for garbage collection. Useful in Jupyter notebooks or tests.

        Note: This is safe to call multiple times as the C++ implementation is idempotent.
        """
        self._cleanup_module(self.pymodule)

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
