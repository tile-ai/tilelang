"""NVRTC Source Wrapper for TileLang.

This module provides C++ kernel launcher generation for the NVRTC backend.
"""
from __future__ import annotations
from typing import Any

from tvm import IRModule
from tvm.target import Target
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper
from tilelang.jit.adapter.utils import (match_declare_kernel, pythonic_expr,
                                        parse_function_call_args, parse_tma_descriptor_args)

PREDEF_HOST_FUNC_PY = """
from cuda.bindings.driver import (
    CUtensorMapDataType,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill,
    cuTensorMapEncodeTiled,
    cuTensorMapEncodeIm2col,
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
    {0}_type = CUtensorMapDataType({1})
    {0}_tensorRank = {2}
    {0}_globalAddress = {3}.data_ptr()
    {0}_globalDim = [{4}]
    {0}_globalStride = [{5}][1:]
    {0}_boxDim = [{6}]
    {0}_elementStrides = [{7}]
    {0}_interleave = CUtensorMapInterleave({8})
    {0}_swizzle = CUtensorMapSwizzle({9})
    {0}_l2Promotion = CUtensorMapL2promotion({10})
    {0}_oobFill = CUtensorMapFloatOOBfill({11})

    res, {0} = cuTensorMapEncodeTiled(
        {0}_type,
        {0}_tensorRank,
        {0}_globalAddress,
        {0}_globalDim,
        {0}_globalStride,
        {0}_boxDim,
        {0}_elementStrides,
        {0}_interleave,
        {0}_swizzle,
        {0}_l2Promotion,
        {0}_oobFill,
    )

    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to initialize the TMA descriptor {0}: {{res}}")
"""

TMA_IM2COL_DESC_INIT_FUNC_PY = """
    {0}_type = CUtensorMapDataType({1})
    {0}_tensorRank = {2}
    {0}_globalAddress = {3}.data_ptr()
    {0}_globalDim = [{4}]
    {0}_globalStride = [{5}][1:]
    {0}_elementStrides = [{6}]
    {0}_lowerCorner = [{7}]
    {0}_upperCorner = [{8}]
    {0}_channelsPerPixel = {9}
    {0}_pixelsPerColumn = {10}
    {0}_interleave = CUtensorMapInterleave({11})
    {0}_swizzle = CUtensorMapSwizzle({12})
    {0}_l2Promotion = CUtensorMapL2promotion({13})
    {0}_oobFill = CUtensorMapFloatOOBfill({14})

    res, {0} = cuTensorMapEncodeIm2col(
        {0}_type,
        {0}_tensorRank,
        {0}_globalAddress,
        {0}_globalDim,
        {0}_globalStride,
        {0}_lowerCorner,
        {0}_upperCorner,
        {0}_channelsPerPixel,
        {0}_pixelsPerColumn,
        {0}_elementStrides,
        {0}_interleave,
        {0}_swizzle,
        {0}_l2Promotion,
        {0}_oobFill,
    )

    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to initialize the TMA descriptor {0}: {{res}}")
"""

L2_PERSISTENT_MAP_CREATE_HANDLE_PY = """
    from cuda.bindings.driver import (
        CUstreamAttrValue,
        CUstreamAttrID,
        CUlimit,
        CUaccessProperty,
        cuCtxGetLimit,
        cuCtxSetLimit,
        cuStreamSetAttribute,
        cuCtxResetPersistingL2Cache,
    )

    stream_attribute = CUstreamAttrValue()
    res, init_persisting_l2_cache_size = cuCtxGetLimit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE)
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get L2 cache size limit: {{res}}")
"""

L2_PERSISTENT_MAP_INIT_FUNC_PY = """
    stream_attribute.accessPolicyWindow.hitRatio = {1}
    stream_attribute.accessPolicyWindow.hitProp = CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING
    stream_attribute.accessPolicyWindow.missProp = CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING

    res = cuCtxSetLimit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE, {2})[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set L2 cache size limit: {{res}}")

    stream_attribute.accessPolicyWindow.base_ptr = {0}.data_ptr()
    stream_attribute.accessPolicyWindow.num_bytes = {2}

    res = cuStreamSetAttribute(stream, CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, stream_attribute)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set stream L2 access policy: {{res}}")
"""

L2_PERSISTENT_MAP_RESET_HANDLE_PY = """
    stream_attribute.accessPolicyWindow.num_bytes = 0
    res = cuStreamSetAttribute(stream, CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, stream_attribute)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to reset stream L2 access policy: {{res}}")

    res = cuCtxResetPersistingL2Cache()[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to reset L2 cache: {{res}}")

    res = cuCtxSetLimit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE, init_persisting_l2_cache_size)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to restore L2 cache size limit: {{res}}")
"""

KERNEL_LAUNCH_FUNC_PY = """
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        {7},
        kernels["{0}"],
        CUdevice({10})
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to {7} for kernel {0}: {{res}}")

    config = CUlaunchConfig()
    config.gridDimX = {1}
    config.gridDimY = {2}
    config.gridDimZ = {3}
    config.blockDimX = {4}
    config.blockDimY = {5}
    config.blockDimZ = {6}
    config.sharedMemBytes = {7}
    config.hStream = stream

    arg_values = {8}
    arg_types = {9}

    res = cuLaunchKernelEx(config, kernels["{0}"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel {0}: {{res}}")
"""


class TLNVRTCSourceWrapper(TLCUDASourceWrapper):
    """
    A wrapper class for the TileLang NVRTC backend.
    """

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

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        return pythonic_expr(expr, dtype_map=None, ignore_cast=True)

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

        # Check if any function needs L2 Persistent Map
        has_l2_persistent_map = False
        for function_name, _ in function_informations.items():
            if function_name in self.l2_persistent_map:
                has_l2_persistent_map = True
                break

        desc_name_map: dict[str, str] = {}
        desc_name_var_map: dict[str, tvm.tir.Var] = {}
        device_index = 0
        kernel_launch_code = """"""
        if has_l2_persistent_map:
            kernel_launch_code += L2_PERSISTENT_MAP_CREATE_HANDLE_PY
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

            # Transform function for NVRTC: returns (arg_value, arg_type) tuples
            def transform_nvrtc_arg(name: str, arg_type: str):
                if arg_type == "ctypes.c_void_p":
                    return (f"{name}.data_ptr()", arg_type)
                return (name, arg_type)

            call_args = parse_function_call_args(declaration, function_args, function_params,
                                                 desc_name_map, desc_name_var_map,
                                                 transform_nvrtc_arg)

            for arg_name, arg_type in call_args:
                if arg_type == "ctypes.c_void_p":
                    device_index = f"{arg_name.replace('.data_ptr()', '')}.device.index"
                    break
            arg_names = ", ".join([arg[0] for arg in call_args])
            arg_types = ", ".join([arg[1] for arg in call_args])
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf

            # Generate L2 persistent map initialization for this function
            init_l2_persistent_map = self.generate_l2_persistent_map(function_name)
            kernel_launch_code += init_l2_persistent_map

            # Generate TMA descriptor initialization and kernel launch code
            kernel_launch_code += self.generate_tma_descriptor_args(
                desc_name_map, desc_name_var_map) + KERNEL_LAUNCH_FUNC_PY.format(
                    function_name, self._pythonic_expr(grid_info[0]),
                    self._pythonic_expr(grid_info[1]), self._pythonic_expr(grid_info[2]),
                    self._pythonic_expr(block_info[0]), self._pythonic_expr(block_info[1]),
                    self._pythonic_expr(block_info[2]), smem_str, arg_names, arg_types,
                    device_index)

            # Reset L2 persistent map after kernel execution
            if has_l2_persistent_map:
                kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE_PY

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC_PY.format(
            repr(list(function_informations.keys())), def_args, kernel_launch_code)
        return host_func

    def generate_l2_persistent_map(self, function_name: str) -> str:
        """Generate L2 persistent cache mapping code for Python NVRTC backend."""
        if function_name not in self.l2_persistent_map:
            return ""
        init_l2_persistent_map = ""
        for buffer_name, (hit_ratio,
                          size_in_bytes) in self.l2_persistent_map[function_name].items():
            # Get persisting_l2_cache_max_size
            from tilelang.carver.arch.driver import get_persisting_l2_cache_max_size
            persisting_l2_cache_max_size = get_persisting_l2_cache_max_size()
            try:
                num_bytes = min(size_in_bytes, persisting_l2_cache_max_size)
            except Exception:
                # as size_in_bytes maybe a symbolic expression
                num_bytes = persisting_l2_cache_max_size
            init_l2_persistent_map += L2_PERSISTENT_MAP_INIT_FUNC_PY.format(
                buffer_name, float(hit_ratio), self._pythonic_expr(num_bytes))

        return init_l2_persistent_map

    def generate_tma_descriptor_args(self, desc_name_map: dict[str, str],
                                     desc_name_var_map: dict[str, tvm.tir.Var]) -> str:
        tma_descriptor_init = ""
        if self.tma_descriptor_args is None:
            return tma_descriptor_init

        # Parse TMA descriptor arguments using the common utility
        parsed_params = parse_tma_descriptor_args(self.tma_descriptor_args, desc_name_map,
                                                  desc_name_var_map, self._pythonic_expr)

        # Generate Python code from parsed parameters
        for params in parsed_params:
            if not params.is_img2col:
                tma_descriptor_init += TMA_DESC_INIT_FUNC_PY.format(
                    params.handle_name, params.dtype, params.tensor_rank, params.global_address,
                    ", ".join(map(lambda x: f"cuuint64_t({x})", params.global_dim)),
                    ", ".join(map(lambda x: f"cuuint64_t({x})", params.global_stride)),
                    ", ".join(map(lambda x: f"cuuint32_t({x})", params.box_dim)),
                    ", ".join(map(lambda x: f"cuuint32_t({x})", params.element_strides)),
                    params.interleave, params.swizzle, params.l2_promotion, params.oob_fill)
            else:
                tma_descriptor_init += TMA_IM2COL_DESC_INIT_FUNC_PY.format(
                    params.handle_name, params.dtype, params.tensor_rank, params.global_address,
                    ", ".join(map(lambda x: f"cuuint64_t({x})", params.global_dim)),
                    ", ".join(map(lambda x: f"cuuint64_t({x})", params.global_stride)),
                    ", ".join(map(lambda x: f"cuuint32_t({x})",
                                  params.element_strides)), ", ".join(params.lower_corner),
                    ", ".join(params.upper_corner), params.smem_box_channel, params.smem_box_pixel,
                    params.interleave, params.swizzle, params.l2_promotion, params.oob_fill)

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
