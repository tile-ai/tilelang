# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from tilelang import tvm as tvm
from typing import Optional, List, Dict, Union
from tvm import IRModule
from tvm.target import Target
from .utils import match_global_kernel, is_cuda_target, is_hip_target, get_annotated_device_mod
import re
import logging

PREDEF_ARRTIBUTE_SET_DYNAMIC_MEMORY = """
    cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});
"""

PREDEF_INIT_FUNC = """
extern "C" void init() {{
    {}
}}
"""

PREDEF_HOST_FUNC = """
extern "C" void call({}) {{
{}
}}
"""


class BaseWrapper(ABC):

    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError


logger = logging.getLogger(__name__)


class TLCUDASourceWrapper(object):
    _TYPE_MAP = {
        "float32": "float",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "e4m3_float8": "__nv_fp8_e4m3",
        "e5m2_float8": "__nv_fp8_e5m2",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uchar": "uint8_t",
    }

    backend = "tl"

    def __init__(self, scheduled_ir_module: IRModule, source: str, target: Target):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.function_name: Optional[str] = None
        self.dynamic_smem_buf: Optional[int] = None
        self.block_info: Union[List[int], Dict] = [1, 1, 1]
        self.grid_info: Union[List[int], Dict] = [1, 1, 1]
        self.parse_source_information()
        self.srcpath: Optional[str] = None
        self.libpath: Optional[str] = None
        self.lib_code: Optional[str] = self.update_lib_code(source)

    def parse_source_information(self):
        device_mod = get_annotated_device_mod(self.mod, self.target)
        assert (len(device_mod.functions) == 1
               ), "Only support one function in the module for static shape kernel."
        for g_var, func in device_mod.functions.items():
            self.function_name = g_var.name_hint
            attrs = func.attrs
            if "dyn_shared_memory_buf" in attrs:
                self.dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        self.block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        self.grid_info["xyz".index(tag[-1])] = extent

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: List[str] = []
        for param in prim_func.params:
            buffer = prim_func.buffer_map[param]
            for dim in buffer.shape:
                if isinstance(dim, tvm.tir.Var) and (dim.name not in dynamic_symbolic_set):
                    dynamic_symbolic_set.append(dim.name)
        return dynamic_symbolic_set

    def get_cuda_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        if self.dynamic_smem_buf is not None:
            call_str = (
                PREDEF_ARRTIBUTE_SET_DYNAMIC_MEMORY.format(self.function_name,
                                                           self.dynamic_smem_buf))
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Find the index of the global kernel function in the code
        index = match_global_kernel(code)
        # Extract the declaration of the function starting from the found index
        declaration = code[index:].split(";")[0]

        function_name = self.function_name
        # Get the CUDA initialization function
        init_func = self.get_cuda_init_func()

        # Locate the opening brace of the function to insert arguments
        index = code.index("{", index)
        function_args = []
        # Populate the function arguments from the primary function's parameters and buffers
        for param in self.prim_func.params:
            buffer = self.prim_func.buffer_map[param]
            function_args.append({
                "name": buffer.name,
                "type": self._TYPE_MAP[buffer.dtype] + "* __restrict__",
            })

        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)
        # Add dynamic symbolic parameters as integers to the function arguments
        for dyn_sym in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append({"name": "stream=cudaStreamDefault", "type": "cudaStream_t"},)
        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s, function_args):
            # Extract the function call arguments matching the function definition
            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        call_args = ", ".join(func_call_args(declaration, function_args))
        block_info, grid_info = self.block_info, self.grid_info

        def legalize_c(p):
            # Convert TIR expressions to legal C expressions
            # Directly convert to string since the special case handling
            # does not alter the string representation for `tvm.tir.Var` and `IntImm`.
            # Replace Python's floor division operator with C's division operator
            if isinstance(p, tvm.tir.IntImm):
                p = int(p)
            return str(p).replace("//", "/")

        # Prepare the block and grid dimensions for the CUDA kernel launch
        block_str = "dim3({}, {}, {})".format(
            legalize_c(block_info[0]),
            legalize_c(block_info[1]),
            legalize_c(block_info[2]),
        )
        grid_str = "dim3({}, {}, {})".format(
            legalize_c(grid_info[0]), legalize_c(grid_info[1]), legalize_c(grid_info[2]))
        # Determine the shared memory size, defaulting to 0 if not specified
        smem_str = 0 if self.dynamic_smem_buf is None else self.dynamic_smem_buf
        # Format the CUDA kernel launch string
        call_str = ""
        if len(dynamic_symbolic_set) != 0:
            call_str += "if ({} == 0) return; \n\t\t".format(list(dynamic_symbolic_set)[0])
        else:
            call_str += ""
        call_str += "{}<<<{}, {}, {}, stream>>>({});".format(function_name, grid_str, block_str,
                                                             smem_str, call_args)
        # Create the host function wrapper for the CUDA kernel
        host_func = PREDEF_HOST_FUNC.format(def_args, call_str)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + init_func + host_func
        return lib_code

    @property
    def prim_func(self):
        if len(self.mod.get_global_vars()) == 1:
            return self.mod[self.mod.get_global_vars()[0]]
        elif "main" in self.mod:
            return self.mod["main"]
        else:
            for _, function in self.mod.functions_items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")


class TLHIPSourceWrapper(TLCUDASourceWrapper):

    def __init__(self, scheduled_ir_module: IRModule, source: str, target: Target):
        super().__init__(scheduled_ir_module, source, target)

    def get_hip_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        if self.dynamic_smem_buf is not None:
            call_str = PREDEF_ARRTIBUTE_SET_DYNAMIC_MEMORY.format(self.function_name,
                                                                  self.dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def get_stream_type(self, function_args):
        function_args.append({"name": "stream=hipStreamDefault", "type": "hipStream_t"},)


class TLWrapper(BaseWrapper):

    def __init__(self, target: Target):
        super().__init__()
        self.scheduled_ir_module = None
        self.target = target
        self.lib = None

    def assign_optimized_module(self, scheduled_ir_module: IRModule):
        self.scheduled_ir_module = scheduled_ir_module

    # Get Scheduled Rt Module and return source to be compiled
    def wrap(self, c_source: str):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target):
            wrapper_class = TLCUDASourceWrapper
        elif is_hip_target(self.target):
            wrapper_class = TLHIPSourceWrapper
        else:
            raise ValueError(f"Unsupported platform: {self.arch.platform}")
        wrapper = wrapper_class(self.scheduled_ir_module, c_source, self.target)
        return wrapper.lib_code
