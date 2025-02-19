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
        dynamic_symbolic_set = set()
        for param in prim_func.params:
            buffer = prim_func.buffer_map[param]
            for dim in buffer.shape:
                if isinstance(dim, tvm.tir.Var):
                    dynamic_symbolic_set.add(dim.name)
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
        if len(dynamic_symbolic_set) != 0:
            call_str = "if ({} == 0) return; \n\t\t".format(list(dynamic_symbolic_set)[0])
        else:
            call_str = ""
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


class TLCUDASourceWrapperWithDynamic(TLCUDASourceWrapper):

    def __init__(self, scheduled_ir_module: IRModule, source: str, target: Target):
        super().__init__(scheduled_ir_module, source, target)

    def get_cuda_init_func(self):
        # Initialize an empty string to accumulate CUDA function calls for setting dynamic shared memory
        call_str = """"""
        # Iterate over functions and their dynamic shared memory requirements
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ARRTIBUTE_SET_DYNAMIC_MEMORY.format(
                    function_name, dynamic_smem_buf)
        # Define the init function that will set the attributes for each kernel
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        # Find the location of the global kernel function in the code
        index = match_global_kernel(code)

        # Analyze the function declaration to prepare for argument extraction
        dummy_declaration = code[index:].split(";")[0]

        function_name = self.function_name

        # Identify the start of the function body to insert arguments
        index = code.index("{", index)
        function_args = []
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            buffer = self.prim_func.buffer_map[param]
            function_args.append({
                "name": buffer.name,
                "type": self._TYPE_MAP[buffer.dtype] + "* __restrict__",
            })
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append({"name": "stream=cudaStreamDefault", "type": "cudaStream_t"},)

        # Format the argument definitions for function declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s: str, function_args):
            # Extract and clean the function call arguments to match the declaration
            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                match = re.sub(r"\d+", "", match)  # Remove numbers
                match = re.sub(r"_", "", match)  # Remove underscores
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        call_args = ", ".join(func_call_args(dummy_declaration, function_args))

        def legalize_c(p):
            # Convert TIR expressions to legal C expressions
            # Directly convert to string since the special case handling
            # does not alter the string representation for `tvm.tir.Var` and `IntImm`.
            # Replace Python's floor division operator with C's division operator
            if isinstance(p, tvm.tir.IntImm):
                p = int(p)
            return str(p).replace("//", "/")

        last_range = 0
        num_items = len(function_informations)
        _call_str = """"""
        for last_range, (function_name, info) in enumerate(function_informations.items()):
            # Prepare block and grid configurations for kernel launches
            block_info, grid_info = info["block_info"], info["grid_info"]
            block_str = "dim3({}, {}, {})".format(
                legalize_c(block_info[0]),
                legalize_c(block_info[1]),
                legalize_c(block_info[2]),
            )
            grid_str = "dim3({}, {}, {})".format(
                legalize_c(grid_info[0]),
                legalize_c(grid_info[1]),
                legalize_c(grid_info[2]),
            )
            # Handle dynamic shared memory specification
            smem_str = (0 if info["dynamic_smem_buf"] is None else info["dynamic_smem_buf"])
            opt_shapes = info["opt_shapes"]
            # Generate conditional kernel launch code based on dynamic symbolic ranges
            (symbolic,) = list(dynamic_symbolic_set)
            range_str = opt_shapes[symbolic]
            if last_range == 0:
                call_str = "  if ({} == 0) return; \n".format(symbolic,)
                call_str += "  if ({} <= {}) {{\n    {}<<<{}, {}, {}, stream>>>({}); \n  }}\n".format(
                    symbolic,
                    range_str,
                    function_name,
                    grid_str,
                    block_str,
                    smem_str,
                    call_args,
                )
            else:
                call_str = "  else if ({} <= {}) {{\n    {}<<<{}, {}, {}, stream>>>({}); \n  }}\n".format(
                    symbolic,
                    range_str,
                    function_name,
                    grid_str,
                    block_str,
                    smem_str,
                    call_args,
                )
            if last_range == num_items - 1:
                call_str += "  else {{\n    {}<<<{}, {}, {}, stream>>>({}); \n  }}\n".format(
                    function_name, grid_str, block_str, smem_str, call_args)
            _call_str += call_str

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC.format(def_args, _call_str)
        return host_func

    def parse_source_information(self):
        # Parse device module to extract execution configurations for each function
        device_mod = get_annotated_device_mod(self.mod, self.target, backend=self.backend)
        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        for g_var, func in device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            function_name = g_var.name_hint
            attrs = func.attrs
            dynamic_smem_buf = None
            if "dyn_shared_memory_buf" in attrs:
                dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                # Extract block and grid sizes from thread extents
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        grid_info["xyz".index(tag[-1])] = extent
            # Map the extracted configurations to each function
            block_info_map[function_name] = block_info
            grid_info_map[function_name] = grid_info
            dynamic_smem_buf_map[function_name] = dynamic_smem_buf
        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map

    def update_lib_code(self, code: str):
        # Organize function information for code generation
        function_informations = {}
        for g_var, func in self.mod.functions.items():
            function_name = g_var.name_hint
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            attrs = func.attrs
            assert "opt_shapes" in attrs
            opt_shapes = attrs["opt_shapes"]
            function_informations[function_name] = {
                "function_name": function_name,
                "opt_shapes": opt_shapes,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
            }

        def compare_map_objects(map_obj):
            comparable_representation = list(map_obj.values())
            return comparable_representation

        function_informations = dict(
            sorted(
                function_informations.items(),
                key=lambda item: compare_map_objects(item[1]["opt_shapes"]),
            ))

        self.lib_code = code

        # Generate the initialization and dispatch functions
        init_func = self.get_cuda_init_func()
        host_func = self.create_dispatch_func(code, function_informations)
        # Concatenate source code with generated code segments
        lib_code = self.source + init_func + host_func
        return lib_code


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
    def wrap(self, c_source: str, is_dynamic: bool = False):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target):
            wrapper_class = (
                TLCUDASourceWrapper if not is_dynamic else TLCUDASourceWrapperWithDynamic)
        elif is_hip_target(self.target):
            wrapper_class = TLHIPSourceWrapper
        else:
            raise ValueError(f"Unsupported platform: {self.arch.platform}")
        wrapper = wrapper_class(self.scheduled_ir_module, c_source, self.target)
        return wrapper.lib_code
