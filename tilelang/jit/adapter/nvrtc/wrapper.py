"""NVRTC Source Wrapper for TileLang.

This module provides C++ kernel launcher generation for the NVRTC backend.
"""
from __future__ import annotations
from typing import Any
import re

from tvm import IRModule
from tvm.target import Target
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper
from tilelang.jit.adapter.utils import match_declare_kernel

PREDEF_HOST_FUNC_PY = """
from cuda.bindings.driver import (
    CUtensorMapDataType,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill,
    cuTensorMapEncodeTiled,
    CUresult,
    cuKernelSetAttribute,
    CUfunction_attribute,
    CUdevice,
    CUlaunchConfig,
    cuLaunchKernelEx,
    cuuint64_t,
    cuuint32_t,
    CUkernel,
)
import ctypes

_function_names = {}

def call({}):
    {}
"""

TMA_DESC_INIT_FUNC_PY = """
\t{0}_type = CUtensorMapDataType({1})
\t{0}_tensorRank = {2}
\t{0}_globalAddress = {3}.data_ptr()
\t{0}_globalDim = [{4}]
\t{0}_globalStride = [{5}][1:]
\t{0}_boxDim = [{6}]
\t{0}_elementStrides = [{7}]
\t{0}_interleave = CUtensorMapInterleave({8})
\t{0}_swizzle = CUtensorMapSwizzle({9})
\t{0}_l2Promotion = CUtensorMapL2promotion({10})
\t{0}_oobFill = CUtensorMapFloatOOBfill({11})

\tres, {0} = cuTensorMapEncodeTiled(
\t\t{0}_type,
\t\t{0}_tensorRank,
\t\t{0}_globalAddress,
\t\t{0}_globalDim,
\t\t{0}_globalStride,
\t\t{0}_boxDim,
\t\t{0}_elementStrides,
\t\t{0}_interleave,
\t\t{0}_swizzle,
\t\t{0}_l2Promotion,
\t\t{0}_oobFill,
\t)

\tif res != CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to initialize the TMA descriptor {0}: {{res}}")
"""

KERNEL_LAUNCH_FUNC_PY = """
\tres = cuKernelSetAttribute(
\t\tCUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
\t\t{7},
\t\tkernels["{0}"],
\t\tCUdevice({10})
\t)[0]
\tif res != CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to set max dynamic shared memory size to {7} for kernel {0}: {{res}}")

\tconfig = CUlaunchConfig()
\tconfig.gridDimX = {1}
\tconfig.gridDimY = {2}
\tconfig.gridDimZ = {3}
\tconfig.blockDimX = {4}
\tconfig.blockDimY = {5}
\tconfig.blockDimZ = {6}
\tconfig.sharedMemBytes = {7}
\tconfig.hStream = stream

\targ_values = {8}
\targ_types = {9}

\tres = cuLaunchKernelEx(config, kernels["{0}"], (arg_values, arg_types), 0)[0]
\tif res != CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to launch kernel {0}: {{res}}")
"""


class TLNVRTCSourceWrapper(TLCUDASourceWrapper):
    """
    A wrapper class for the TileLang NVRTC backend.
    """

    _TYPE_MAP = {
        "float32": "ctypes.c_float",
        "float16": "ctypes.c_uint16",
        "bfloat16": "ctypes.c_uint16",
        "float8_e4m3": "ctypes.c_uint8",
        "float8_e4m3fn": "ctypes.c_uint8",
        "float8_e5m2": "ctypes.c_uint8",
        "float64": "ctypes.c_double",
        "int64": "ctypes.c_int64",
        "int32": "ctypes.c_int32",
        "uint32": "ctypes.c_uint32",
        "bool": "ctypes.c_bool",
        "int8": "ctypes.c_int8",
        "uint8": "ctypes.c_uint8",
        "int16": "ctypes.c_int16",
        "uint16": "ctypes.c_uint16",
        "uchar": "ctypes.c_uint8",
    }

    _generated_host_func: str | None = None

    def __init__(self,
                 scheduled_ir_module: IRModule,
                 source: str,
                 target: Target,
                 device_mod: IRModule | None = None,
                 host_mod: IRModule | None = None,
                 pass_configs: dict[str, Any] | None = None):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    @property
    def host_func(self):
        """Override parent's host_func to return generated Python code."""
        if self._generated_host_func is not None:
            return self._generated_host_func
        return super().host_func

    @host_func.setter
    def host_func(self, value):
        """Allow setting generated host function code."""
        self._generated_host_func = value

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = [{"name": "kernels", "type": "dict[str, CUkernel]"}]
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({
                    "name": buffer.data.name,
                    "type": "ctypes.c_void_p",
                })
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(
                    f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": "ctypes.c_int"})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['name']}" for arg in function_args])

        def func_call_args(s,
                           function_args,
                           function_params,
                           desc_name_map: dict[str, str] | None = None,
                           desc_name_var_map: dict[str, tvm.tir.Var] | None = None,
                           device_index: int | None = None):
            # Extract the function call arguments matching the function definition
            def maybe_desc(name: str, matches: list[str], i: int):
                match = matches[i]
                if not (match == name + "_desc" or match.startswith(name + "_desc_")):
                    return False
                desc_decls = []
                if desc_name_map is not None:
                    desc_name_map[match] = name
                if i > 0:
                    desc_decls.append(matches[i - 1])
                if i < len(matches) - 1:
                    desc_decls.append(matches[i + 1])
                return any([decl == "CUtensorMap" for decl in desc_decls])

            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for i, match in enumerate(matches):
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(
                            (f"{match}.data_ptr()" if arg["type"] == "ctypes.c_void_p" else match,
                             arg["type"]))
                    elif maybe_desc(arg["name"], matches, i):
                        call_args.append((match, "None"))
                        assert len(call_args) <= len(
                            function_params
                        ), f"Function {function_name} has {len(function_params)} parameters, but {len(call_args)} arguments"
                        desc_name_var_map[match] = function_params[len(call_args) - 1]
            return call_args

        # [TODO] L2 Persistent Map

        desc_name_map: dict[str, str] = {}
        desc_name_var_map: dict[str, tvm.tir.Var] = {}
        device_index = 0
        kernel_launch_code = """"""
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)
            call_args = func_call_args(declaration, function_args, function_params, desc_name_map,
                                       desc_name_var_map)
            for arg_name, arg_type in call_args:
                if arg_type == "ctypes.c_void_p":
                    device_index = f"{arg_name.replace('.data_ptr()', '')}.device.index"
                    break
            arg_names = ", ".join([arg[0] for arg in call_args])
            arg_types = ", ".join([arg[1] for arg in call_args])
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
            kernel_launch_code += self.generate_tma_descriptor_args(
                desc_name_map, desc_name_var_map) + KERNEL_LAUNCH_FUNC_PY.format(
                    function_name, self._pythonic_expr(grid_info[0]),
                    self._pythonic_expr(grid_info[1]), self._pythonic_expr(grid_info[2]),
                    self._pythonic_expr(block_info[0]), self._pythonic_expr(block_info[1]),
                    self._pythonic_expr(block_info[2]), smem_str, arg_names, arg_types,
                    device_index)

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC_PY.format(
            repr(list(function_informations.keys())), def_args, kernel_launch_code)
        return host_func

    def generate_tma_descriptor_args(self, desc_name_map: dict[str, str],
                                     desc_name_var_map: dict[str, tvm.tir.Var]) -> str:
        tma_descriptor_init = ""
        if self.tma_descriptor_args is None:
            return tma_descriptor_init

        for handle_name, _ in desc_name_map.items():
            assert handle_name in desc_name_var_map, f"Handle name {handle_name} not found in desc_name_var_map"
            desc_var = desc_name_var_map[handle_name]

            assert desc_var in self.tma_descriptor_args, f"TMA descriptor {desc_var} not found in {self.tma_descriptor_args}"
            args = self.tma_descriptor_args[desc_var]
            # Skip __tvm_tensormap_create_tiled
            if len(args) < 3:
                raise ValueError(
                    f"TMA descriptor args too short: {len(args)} elements, expected at least 3")
            _, dtype, tensor_rank, globalAddress, *remaining_args = args[1:]

            tensor_rank = int(tensor_rank)
            # Validate tensor_rank
            if not isinstance(tensor_rank, int) or tensor_rank <= 0:
                raise ValueError(f"Invalid tensor_rank: {tensor_rank}. Must be a positive integer")

            # Calculate required length for remaining_args
            # 4 groups of tensor_rank size + 4 parameters
            expected_args_len = 4 * tensor_rank + 4
            if len(remaining_args) < expected_args_len:
                raise ValueError(f"Insufficient remaining args: got {len(remaining_args)}, "
                                 f"expected {expected_args_len} for tensor_rank {tensor_rank}")

            # Extract dimensions and strides using list slicing
            global_dim = remaining_args[:tensor_rank]
            global_stride = remaining_args[tensor_rank:2 * tensor_rank]
            box_dim = remaining_args[2 * tensor_rank:3 * tensor_rank]
            element_strides = remaining_args[3 * tensor_rank:4 * tensor_rank]

            global_dim = [str(i) for i in global_dim]
            global_stride = [str(i) for i in global_stride]
            box_dim = [str(i) for i in box_dim]
            element_strides = [str(i) for i in element_strides]

            # Extract remaining parameters
            try:
                interleave, swizzle, l2Promotion, oobFill = remaining_args[4 * tensor_rank:4 *
                                                                           tensor_rank + 4]
            except ValueError as e:
                raise ValueError(
                    "Failed to unpack the final 4 TMA parameters (interleave, swizzle, l2Promotion, oobFill)"
                ) from e

            tma_descriptor_init += TMA_DESC_INIT_FUNC_PY.format(
                handle_name, dtype, tensor_rank, globalAddress,
                ", ".join(map(lambda x: f"cuuint64_t({x})", global_dim)),
                ", ".join(map(lambda x: f"cuuint64_t({x})", global_stride)),
                ", ".join(map(lambda x: f"cuuint32_t({x})", box_dim)),
                ", ".join(map(lambda x: f"cuuint32_t({x})",
                              element_strides)), interleave, swizzle, l2Promotion, oobFill)
        return tma_descriptor_init

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code

        # Organize function information for code generation
        function_informations = {}
        for function_name in self.function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] | None = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tir.Call):
                    if not (hasattr(node, "op") and
                            node.op == tvm.ir.Op.get("tir.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError(
                            "tvm_call_packed should have at least 1 argument and match device function parameters"
                        )
                    function_params = args[1:1 + param_cnt]

            post_order_visit(self.host_func.body, visitor)
            assert function_params is not None, "function_params should not be None"

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
                "function_params": function_params,
            }

        # Create the host function wrapper for the CUDA kernel
        self.host_func = self.create_dispatch_func(code, function_informations)
        return self.lib_code

    def get_stream_type(self) -> dict[str, str]:
        return {"name": "stream=0", "type": "int"}
