# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

import torch
from ..base import BaseKernelAdapter
import ctypes
from typing import List, Optional, Union, Callable, Dict, Tuple
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.relay import TensorType
from tvm import tir
from .wrapper import TLWrapper
from .libgen import LibraryGenerator
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module


class CtypesKernelAdapter(BaseKernelAdapter):

    target = "cuda"
    ir_module = None
    lib: Optional[ctypes.CDLL] = None
    wrapped_source: Optional[str] = None
    # SymbolicVar: (Buffer Index, Shape Index)
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None

    def __init__(self,
                 rt_mod,
                 params: List[TensorType],
                 result_idx: List[int],
                 target,
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 verbose: bool = False):

        self.mod = rt_mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        self.target = Target.canon_target(determine_target(target))
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source())

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
        self.lib.init()

        self._post_init()

    def _process_dynamic_symbolic(self):
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

    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def _warp_forward_from_prebuild_lib(self,
                                        *ins: List[torch.Tensor],
                                        stream: Optional[int] = None):
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )
        ins_idx = 0
        args = []

        # tensor pointers
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = torch.__getattribute__(str(self.params[i].dtype))
                shape = list(map(int, self.params[i].shape))
                # use the device of the first input tensor if available
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
            stream = torch.cuda.current_stream().cuda_stream

        self._forward_from_prebuild_lib(*args, stream=stream)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        else:
            return [args[i] for i in self.result_idx]

    def _convert_torch_func(self) -> Callable:
        return self._warp_forward_from_prebuild_lib

    @property
    def prim_func(self) -> tir.PrimFunc:
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        return self.lib_generator.srcpath

    @property
    def libpath(self):
        return self.lib_generator.libpath

    @property
    def lib_code(self):
        return self.lib_generator.lib_code

    @property
    def is_dynamic(self):
        return (self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0)
