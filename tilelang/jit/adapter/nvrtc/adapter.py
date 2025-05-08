# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

from ..base import BaseKernelAdapter
from typing import List, Union, Callable, Optional, Dict, Any
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.target import determine_target
import logging
from tilelang.contrib.nvrtc import compile_cuda
import os
import os.path as osp
import tempfile
import sys
import uuid
import importlib
import cuda.bindings.driver as cuda
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
        
        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
        
        self.result_idx = result_idx
        self.target = Target.canon_target(determine_target(target))
        self.wrapper = TLPyWrapper(self.target)
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapper_source, self.host_func, self.function_names = self.wrapper.wrap(kernel_global_source)
        
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

        cubin_bytes = compile_cuda(self.wrapper_source, 
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
        self.func = self._convert_torch_func()
        
    def get_kernel_source(self):
        return self.wrapper_source

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""
        def lambda_forward(*args, stream: int = -1):
            if stream == -1:
                return self.lib.call(self.kernels, *args)
            else:
                return self.lib.call(self.kernels, *args, stream=stream)

        return lambda_forward
