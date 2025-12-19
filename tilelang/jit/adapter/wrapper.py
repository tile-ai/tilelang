from __future__ import annotations
from abc import ABC, abstractmethod
from tilelang import tvm as tvm
from typing import Any
from tvm import IRModule
from tvm.target import Target

from .utils import (
    is_metal_target,
    is_cutedsl_target,
    match_declare_kernel,
    match_declare_kernel_cpu,
    is_cuda_target,
    is_hip_target,
    is_cpu_target,
    get_annotated_mod,
    pythonic_expr,
    parse_function_call_args,
    parse_tma_descriptor_args,
)
import re
import logging
import textwrap
from tvm.tir.stmt_functor import post_order_visit

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY = """
    cudaError_t result_{0} = cudaFuncSetAttribute({0}, cudaFuncAttributeMaxDynamicSharedMemorySize, {1});
    if (result_{0} != cudaSuccess) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", {1}, cudaGetErrorString(result_{0}));
        return -1;
    }}
"""

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP = """
    if ({1} > 65536) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size for {0} to %d", {1});
        return -1;
    }}
    return 0;
"""

PREDEF_INIT_FUNC = """
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {{
    return error_buf;
}}

extern "C" int init() {{
    error_buf[0] = '\\0';
    {0}
    return 0;
}}
"""

PREDEF_HOST_FUNC = """
extern "C" int call({}) {{
{}
\treturn 0;
}}
"""

L2_PERSISTENT_MAP_CREATE_HANDLE = """
\tcudaStreamAttrValue stream_attribute;
\tsize_t init_persisting_l2_cache_size;
\tcudaDeviceGetLimit(&init_persisting_l2_cache_size, cudaLimitPersistingL2CacheSize);
"""

L2_PERSISTENT_MAP_INIT_FUNC = """
\tstream_attribute.accessPolicyWindow.hitRatio = {1};
\tstream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
\tstream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
\tcudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, {2});
\tstream_attribute.accessPolicyWindow.base_ptr = (void*)({0});
\tstream_attribute.accessPolicyWindow.num_bytes = {2};
\tcudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
"""

L2_PERSISTENT_MAP_RESET_HANDLE = """
\tstream_attribute.accessPolicyWindow.num_bytes = 0;
\tcudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
\tcudaCtxResetPersistingL2Cache();
\tcudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, init_persisting_l2_cache_size);
"""

TMA_DESC_INIT_FUNC = """
\tCUtensorMap {0};
\tCUtensorMapDataType {0}_type= (CUtensorMapDataType){1};
\tcuuint32_t {0}_tensorRank= {2};
\tvoid *{0}_globalAddress= {3};
\tcuuint64_t {0}_globalDim[{2}]= {{{4}}};
\tcuuint64_t {0}_globalStride[{2}]= {{{5}}};
\tcuuint32_t {0}_boxDim[{2}]= {{{6}}};
\tcuuint32_t {0}_elementStrides[{2}]= {{{7}}};
\tCUtensorMapInterleave {0}_interleave= (CUtensorMapInterleave){8};
\tCUtensorMapSwizzle {0}_swizzle= (CUtensorMapSwizzle){9};
\tCUtensorMapL2promotion {0}_l2Promotion= (CUtensorMapL2promotion){10};
\tCUtensorMapFloatOOBfill {0}_oobFill= (CUtensorMapFloatOOBfill){11};

\tCUresult {0}_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &{0}, {0}_type, {0}_tensorRank, {0}_globalAddress, {0}_globalDim, {0}_globalStride + 1, {0}_boxDim, {0}_elementStrides, {0}_interleave, {0}_swizzle, {0}_l2Promotion, {0}_oobFill);

\tif ({0}_result != CUDA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TMA descriptor {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}
"""

TMA_IM2COL_DESC_INIT_FUNC = """
\tCUtensorMap {0};
\tCUtensorMapDataType {0}_type= (CUtensorMapDataType){1};
\tcuuint32_t {0}_tensorRank= {2};
\tvoid *{0}_globalAddress= {3};
\tcuuint64_t {0}_globalDim[{2}]= {{{4}}};
\tcuuint64_t {0}_globalStride[{2}]= {{{5}}};
\tcuuint32_t {0}_elementStrides[{2}]= {{{6}}};
\tint {0}_lowerCorner[{2} - 2]= {{{7}}};
\tint {0}_upperCorner[{2} - 2]= {{{8}}};
\tcuuint32_t {0}_channelsPerPixel= {9};
\tcuuint32_t {0}_pixelsPerColumn= {10};
\tCUtensorMapInterleave {0}_interleave= (CUtensorMapInterleave){11};
\tCUtensorMapSwizzle {0}_swizzle= (CUtensorMapSwizzle){12};
\tCUtensorMapL2promotion {0}_l2Promotion= (CUtensorMapL2promotion){13};
\tCUtensorMapFloatOOBfill {0}_oobFill= (CUtensorMapFloatOOBfill){14};

\tCUresult {0}_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeIm2col)(
    &{0}, {0}_type, {0}_tensorRank, {0}_globalAddress, {0}_globalDim, {0}_globalStride + 1,
    {0}_lowerCorner, {0}_upperCorner, {0}_channelsPerPixel, {0}_pixelsPerColumn, {0}_elementStrides, {0}_interleave, {0}_swizzle, {0}_l2Promotion, {0}_oobFill);

\tif ({0}_result != CUDA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TMA descriptor {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}
"""


class BaseWrapper(ABC):
    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError


logger = logging.getLogger(__name__)


class TLCUDASourceWrapper:
    _TYPE_MAP = {
        "float32": "float",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e4m3fn": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "uchar": "uint8_t",
    }

    backend = "tl"
    device_mod: IRModule | None = None
    host_mod: IRModule | None = None
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.function_names: str | None = None
        self.dynamic_smem_buf: int | None = None
        self.block_info: list[int] | dict = [1, 1, 1]
        self.grid_info: list[int] | dict = [1, 1, 1]
        self.tma_descriptor_args: dict | None = None
        self.l2_persistent_map: dict[str, dict] | None = {}
        self.parse_source_information()
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = self.update_lib_code(source)

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        # This wrapper generates C/CUDA source. C/C++ integer division uses '/',
        # and '//' is not a valid operator in C/C++.
        return pythonic_expr(expr, self._TYPE_MAP, floor_div_op="/")

    def _lookup_type(self, dtype: str | Any) -> str:
        key = dtype if isinstance(dtype, str) else str(dtype)
        result = self._TYPE_MAP.get(key)
        assert result is not None, f"Unsupported dtype {dtype}"
        return result

    def is_tma_descriptor_arg(self, arg_name: str) -> bool:
        return arg_name in self.prim_func.buffer_map

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []

        # Collect function arguments based on primary function's parameters and buffer mappings
        # QA(@lei): Why not use device_mod.params?
        # device func lack buffer map (to convert buffer handle to buffer)
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append(
                    {
                        "name": buffer.data.name,
                        "type": self._lookup_type(buffer.dtype) + "* __restrict__",
                    }
                )
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": self._lookup_type(dyn_sym_dtype)})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        has_l2_persistent_map = False
        for function_name, _ in function_informations.items():
            if function_name in self.l2_persistent_map:
                has_l2_persistent_map = True
                break

        kernel_launch_code = """"""
        if has_l2_persistent_map:
            kernel_launch_code += L2_PERSISTENT_MAP_CREATE_HANDLE
        desc_name_map: dict[str, str] = {}
        desc_name_var_map: dict[str, tvm.tir.Var] = {}
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

            block_str = (
                f"dim3({self._pythonic_expr(block_info[0])}, {self._pythonic_expr(block_info[1])}, {self._pythonic_expr(block_info[2])})"
            )
            grid_str = (
                f"dim3({self._pythonic_expr(grid_info[0])}, {self._pythonic_expr(grid_info[1])}, {self._pythonic_expr(grid_info[2])})"
            )
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
            init_l2_persistent_map = self.generate_l2_persistent_map(function_name)
            kernel_launch_code += init_l2_persistent_map

            if self.use_cooperative_groups[function_name]:
                args_list = parse_function_call_args(declaration, function_args, function_params, desc_name_map, desc_name_var_map)
                assert len(function_params) == len(args_list), (
                    f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
                )
                args_array = [f"(void*)&{arg}" for arg in args_list]
                call_args = f"\tvoid* {function_name}_args[] = {{{', '.join(args_array)}}};\n"
                kernel_launch_code += call_args
                # Using cudaLaunchCooperativeKernel to launch the kernel
                kernel_launch_code += "\tTILELANG_CHECK(cudaLaunchCooperativeKernel((void*){}, {}, {}, {}, {}, stream));\n".format(
                    function_name, grid_str, block_str, function_name + "_args", smem_str
                )
            else:
                args_list = parse_function_call_args(declaration, function_args, function_params, desc_name_map, desc_name_var_map)
                assert len(function_params) == len(args_list), (
                    f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
                )
                call_args = ", ".join(args_list)
                kernel_launch_code += f"\t{function_name}<<<{grid_str}, {block_str}, {smem_str}, stream>>>({call_args});\n"
                kernel_launch_code += f'\tTILELANG_CHECK_LAST_ERROR("{function_name}");\n'
            if has_l2_persistent_map:
                kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE

        init_tma_descriptor_args = self.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
        kernel_launch_code = init_tma_descriptor_args + kernel_launch_code

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        return host_func

    def generate_l2_persistent_map(self, function_name: str) -> str:
        if function_name not in self.l2_persistent_map:
            return ""
        init_l2_persistent_map = ""
        for buffer_name, (hit_ratio, size_in_bytes) in self.l2_persistent_map[function_name].items():
            # get persisting_l2_cache_max_size
            from tilelang.carver.arch.driver import get_persisting_l2_cache_max_size

            persisting_l2_cache_max_size = get_persisting_l2_cache_max_size()
            try:
                num_bytes = min(size_in_bytes, persisting_l2_cache_max_size)
            except Exception:
                # as size_in_bytes maybe a symbolic expression
                num_bytes = persisting_l2_cache_max_size
            init_l2_persistent_map += L2_PERSISTENT_MAP_INIT_FUNC.format(buffer_name, float(hit_ratio), self._pythonic_expr(num_bytes))

        return init_l2_persistent_map

    def generate_tma_descriptor_args(self, desc_name_map: dict[str, str], desc_name_var_map: dict[str, tvm.tir.Var]) -> str:
        tma_descriptor_init = ""
        if self.tma_descriptor_args is None:
            return tma_descriptor_init

        # Parse TMA descriptor arguments using the common utility
        parsed_params = parse_tma_descriptor_args(self.tma_descriptor_args, desc_name_map, desc_name_var_map, self._pythonic_expr)

        # Generate C++ code from parsed parameters
        for params in parsed_params:
            if not params.is_img2col:
                tma_descriptor_init += TMA_DESC_INIT_FUNC.format(
                    params.handle_name,
                    params.dtype,
                    params.tensor_rank,
                    params.global_address,
                    ",".join(params.global_dim),
                    ",".join(params.global_stride),
                    ",".join(params.box_dim),
                    ",".join(params.element_strides),
                    params.interleave,
                    params.swizzle,
                    params.l2_promotion,
                    params.oob_fill,
                )
            else:
                tma_descriptor_init += TMA_IM2COL_DESC_INIT_FUNC.format(
                    params.handle_name,
                    params.dtype,
                    params.tensor_rank,
                    params.global_address,
                    ",".join(params.global_dim),
                    ",".join(params.global_stride),
                    ",".join(params.element_strides),
                    ",".join(params.lower_corner),
                    ",".join(params.upper_corner),
                    params.smem_box_channel,
                    params.smem_box_pixel,
                    params.interleave,
                    params.swizzle,
                    params.l2_promotion,
                    params.oob_fill,
                )

        return tma_descriptor_init

    def parse_source_information(self):
        if self.device_mod is None or self.host_mod is None:
            with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
                device_mod, host_mod = get_annotated_mod(self.mod, self.target)
            self.device_mod = device_mod
            self.host_mod = host_mod
        assert len(self.device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(self.host_mod.functions) == 1, "Only support one function in host module."

        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        function_names = []
        use_cooperative_groups_map = {}
        for g_var, func in self.device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            function_name = g_var.name_hint
            attrs = func.attrs
            dynamic_smem_buf = None
            use_cooperative_groups = False
            if "use_cooperative_groups" in attrs:
                use_cooperative_groups = attrs["use_cooperative_groups"]
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
            use_cooperative_groups_map[function_name] = use_cooperative_groups
            function_names.append(function_name)

        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map
        self.use_cooperative_groups = use_cooperative_groups_map

        function_names_index = {}
        for _, func in self.host_mod.functions.items():
            if "tma_descriptor_args" in func.attrs:
                self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
            if "l2_persistent_map" in func.attrs:
                self.l2_persistent_map[function_name] = func.attrs["l2_persistent_map"]

            host_code = str(func)
            for function_name in function_names:
                index = host_code.index(f'T.call_packed("{function_name}"')
                function_names_index[function_name] = index
        # sort function_names
        function_names = sorted(function_names, key=lambda x: function_names_index[x])
        self.function_names = function_names

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: dict[str, str] = {}

        def unique_push_back(name: str, dtype: str):
            if name not in dynamic_symbolic_set:
                dynamic_symbolic_set[name] = dtype
            else:
                assert dtype == dynamic_symbolic_set[name]

        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for dim in buffer.shape:
                    if isinstance(dim, tvm.tir.Var):
                        unique_push_back(dim.name, str(dim.dtype))

        # Note: In buffer definitions, any dynamic symbols appearing in strides are listed after those in the shape.
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for stride in buffer.strides:
                    if isinstance(stride, tvm.tir.Var):
                        unique_push_back(stride.name, str(stride.dtype))

        return list(dynamic_symbolic_set.items())

    def get_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY.format(function_name, dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Get the function names
        function_names = self.function_names
        # Get the CUDA initialization function
        init_func = self.get_init_func()

        # Organize function information for code generation
        function_informations = {}
        for function_name in function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue
            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tir.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tir.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                    function_params = args[1 : 1 + param_cnt]

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
        host_func = self.create_dispatch_func(code, function_informations)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + init_func + host_func
        return lib_code

    def get_stream_type(self) -> dict[str, str]:
        return {"name": "stream=cudaStreamDefault", "type": "cudaStream_t"}

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

    @property
    def device_func(self):
        if len(self.device_mod.get_global_vars()) == 1:
            return self.device_mod[self.device_mod.get_global_vars()[0]]
        elif "main" in self.device_mod:
            return self.device_mod["main"]
        else:
            for _, function in self.device_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    @property
    def host_func(self):
        if len(self.host_mod.get_global_vars()) == 1:
            return self.host_mod[self.host_mod.get_global_vars()[0]]
        elif "main" in self.host_mod:
            return self.host_mod["main"]
        else:
            for _, function in self.host_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")


class TLHIPSourceWrapper(TLCUDASourceWrapper):
    """
    A wrapper class for the TileLang HIP backend.
    """

    _TYPE_MAP = {
        "float32": "float",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e4m3fn": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float8_e4m3fnuz": "fp8_e4_t",
        "e4m3fnuz_float8": "fp8_e4_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "uchar": "uint8_t",
    }

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    def get_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP.format(function_name, dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def get_stream_type(self) -> dict[str, str]:
        return {"name": "stream=hipStreamDefault", "type": "hipStream_t"}


class TLCPUSourceWrapper:
    _TYPE_MAP = {
        "float32": "float",
        "float16": "half",
        "int32": "int32_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int64": "int64_t",
        "uint64": "uint64_t",
        "float64": "double",
        "bool": "bool",
        "uchar": "uchar",
    }

    # Use common init with error buffer and get_last_error for CPU backend as well
    INIT_FUNC = PREDEF_INIT_FUNC.format("")

    CALL_PREFIX = textwrap.dedent("""
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t call({}) {{
          return {};
        }}
    """)

    backend = "tl"
    device_mod: IRModule | None = None
    host_mod: IRModule | None = None
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.pass_configs = pass_configs
        self.function_names: str | None = None
        self.dynamic_smem_buf: int | None = None
        self.parse_source_information()
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = self.update_lib_code(source)

    def _lookup_type(self, dtype: str | Any) -> str:
        key = dtype if isinstance(dtype, str) else str(dtype)
        result = self._TYPE_MAP.get(key)
        assert result is not None, f"Unsupported dtype {dtype}"
        return result

    def create_call_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append(
                    {
                        "name": buffer.name,
                        "type": self._lookup_type(buffer.dtype) + "*",
                    }
                )
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": self._lookup_type(dyn_sym_dtype)})
        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s, function_args):
            pattern = r"[,\s]*(?:\w+\s*\*+\s*\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        _call_str = """"""

        for function_name, _ in function_informations.items():
            # Find the location of the global kernel function in the code
            index = match_declare_kernel_cpu(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)

            call_args = ", ".join(func_call_args(declaration, function_args))
            _call_str += f"{function_name}({call_args})"

        # Wrap the kernel dispatch logic in an external C function
        host_func = self.CALL_PREFIX.format(def_args, _call_str)
        return host_func

    def parse_source_information(self):
        with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
            device_mod, host_mod = get_annotated_mod(self.mod, self.target)
        assert len(device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(host_mod.functions) == 1, "Only support one function in host module."

        function_names = []
        for g_var, _ in device_mod.functions.items():
            function_name = g_var.name_hint
            function_names.append(function_name)

        self.function_names = function_names

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: dict[str, str] = {}
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for dim in buffer.shape:
                    if isinstance(dim, tvm.tir.Var) and (dim.name not in dynamic_symbolic_set):
                        dynamic_symbolic_set[dim.name] = str(dim.dtype)
        return list(dynamic_symbolic_set.items())

    def get_cpu_init_func(self):
        # Provide init() and get_last_error() for CPU backend
        return self.INIT_FUNC

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Get the function names
        function_names = self.function_names
        # Get the CPU initialization function
        init_func = self.get_cpu_init_func()

        # Organize function information for code generation
        function_informations = {}
        for function_name in function_names:
            function_informations[function_name] = {
                "function_name": function_name,
            }

        # Create the call function wrapper for the CPU kernel
        call_func = self.create_call_func(code, function_informations)
        # Combine the source, initialization function, and call function to form the complete library code
        lib_code = self.source + init_func + call_func
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


class TLMetalSourceWrapper:
    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.lib_code = self.update_lib_code(source)

    def update_lib_code(self, code: str):
        self.lib_code = code
        return self.lib_code


# TLCuTeDSLSourceWrapper has been moved to tilelang.jit.adapter.cutedsl.wrapper


class TLWrapper(BaseWrapper):
    """
    A wrapper class for the TileLang backend.
    """

    device_mod: IRModule | None = None
    host_mod: IRModule | None = None
    pass_configs: dict[str, Any] | None = None
    target: Target | None = None
    lib: object | None = None

    def __init__(self, target: Target):
        super().__init__()
        self.scheduled_ir_module = None
        self.pass_configs = None
        self.target = target
        self.lib = None

    def assign_optimized_module(self, scheduled_ir_module: IRModule):
        self.scheduled_ir_module = scheduled_ir_module

    def assign_pass_configs(self, pass_configs: dict[str, Any]):
        self.pass_configs = pass_configs

    def assign_host_module(self, host_mod: IRModule):
        self.host_mod = host_mod

    def assign_device_module(self, device_mod: IRModule):
        self.device_mod = device_mod

    # Get Scheduled Rt Module and return source to be compiled
    def wrap(self, c_source: str):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target):
            wrapper_class = TLCUDASourceWrapper
        elif is_hip_target(self.target):
            wrapper_class = TLHIPSourceWrapper
        elif is_cpu_target(self.target):
            wrapper_class = TLCPUSourceWrapper
        elif is_metal_target(self.target):
            wrapper_class = TLMetalSourceWrapper
        else:
            raise ValueError(f"Unsupported platform: {self.arch.platform}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=c_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return wrapper.lib_code


class TLPyWrapper(TLWrapper):
    def __init__(self, target: Target):
        super().__init__(target)

    def wrap(self, py_source: str):
        # assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cutedsl_target(self.target):
            from tilelang.jit.adapter.cutedsl import TLCuTeDSLSourceWrapper

            wrapper_class = TLCuTeDSLSourceWrapper
        elif is_cuda_target(self.target):
            from tilelang.jit.adapter.nvrtc import TLNVRTCSourceWrapper

            wrapper_class = TLNVRTCSourceWrapper
        else:
            raise ValueError(f"Unsupported target for NVRTC backend: {self.target}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=py_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return {
            "host_func": getattr(wrapper, "host_func", None),
            "function_names": getattr(wrapper, "function_names", None),
            "tma_cpp_init_code": getattr(wrapper, "tma_cpp_init_code", None),
            "tma_lib_name": getattr(wrapper, "tma_lib_name", None),
            "launcher_cpp_code": getattr(wrapper, "launcher_cpp_code", None),
            "launcher_lib_name": getattr(wrapper, "launcher_lib_name", None),
        }
