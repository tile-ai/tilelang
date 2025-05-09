# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

import importlib
import logging
import os
import os.path as osp
import sys
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

import cuda.bindings.driver as cuda
import torch
from tvm import tir
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.contrib.nvrtc import compile_cuda
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.target import determine_target

from ..base import BaseKernelAdapter

logger = logging.getLogger(__name__)


class NVRTCKernelAdapter(BaseKernelAdapter):
    def __init__(
            self,
            params: List[KernelParam],
            result_idx: List[int],
            target: Union[str, Target],
            func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
            host_mod: Optional[tvm.IRModule] = None,
            device_mod: Optional[tvm.IRModule] = None,
            kernel_global_source: Optional[str] = None,
            verbose: bool = False,
            pass_configs: Optional[Dict[str, Any]] = None):
        
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source
        
        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
            
        # Cache parameter information during initialization
        self.param_dtypes = [param.dtype for param in params]
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
        self.host_func, self.function_names = self.wrapper.wrap(kernel_global_source)
        
        project_root = osp.join(osp.dirname(__file__), "../../..")
        if "TL_TEMPLATE_PATH" in os.environ:
            tl_template_path = os.environ["TL_TEMPLATE_PATH"]
        else:
            tl_template_path = osp.abspath(osp.join(project_root, "src"))
            
        if "TL_CUTLASS_PATH" in os.environ:
            cutlass_path = os.environ["TL_CUTLASS_PATH"]
        else:
            cutlass_path = osp.abspath(
                osp.join(project_root, "3rdparty/cutlass/include"))
            
        if "CUDA_HOME" in os.environ:
            cuda_home = os.environ["CUDA_HOME"]
        else:
            cuda_home = "/usr/local/cuda"

        cubin_bytes = compile_cuda(self.kernel_global_source, 
                                   target_format="cubin", 
                                   options=[f"-I{tl_template_path}", f"-I{cutlass_path}", f"-I{cuda_home}/include"],
                                   verbose=True)
        lib = tempfile.NamedTemporaryFile(mode="wb", suffix=".cubin", delete=False)
        lib.write(cubin_bytes)
        lib.close()
        self.libpath = lib.name
        
        host_dir = tempfile.mkdtemp()
        sys.path.append(host_dir)
        lib_name = str(uuid.uuid4())
        os.makedirs(osp.join(host_dir, lib_name), exist_ok=True)
        with open(osp.join(host_dir, lib_name, "__init__.py"), "w") as f:
            f.write(self.host_func)
        self.lib = importlib.import_module(lib_name)
        
        self.kernels = {}
        result, self.culib = cuda.cuLibraryLoadFromFile(bytes(self.libpath, "utf-8"), [], [], 0, [], [], 0)
        assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to load library: {self.libpath}"
        for name in self.function_names:
            result, self.kernels[name] = cuda.cuLibraryGetKernel(self.culib, bytes(name, "utf-8"))
            assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to get kernel: {name}"
        
        self._post_init()
    
    def _process_dynamic_symbolic(self):
        """Extract information about dynamic shapes from the TIR function.
        
        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
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

        
    def get_kernel_source(self):
        return self.kernel_global_source
    
    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        """Low-level function to call the compiled CUDA kernel.
        """
        return self.lib.call(self.kernels, *args, stream=stream)
    
    def _wrap_forward_from_prebuild_lib(self,
                                        *ins: List[torch.Tensor],
                                        stream: Optional[int] = None):
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
                device = ins[0].device if len(
                    ins) > 0 else torch.cuda.current_device()
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

    def _convert_torch_func(self) -> Callable:
        return self._wrap_forward_from_prebuild_lib
    
    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    def __del__(self):
        result = cuda.cuLibraryUnload(self.culib)[0]
        if result != cuda.CUresult.CUDA_SUCCESS:
            logger.warning(f"Failed to unload library: {self.libpath}")
