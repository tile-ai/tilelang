from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tilelang import tvm as tvm
from typing import Any
from tvm import IRModule
from tvm.target import Target

from .utils import (
    is_ascend_target,
    is_metal_target,
    is_cutedsl_target,
    is_pto_target,
    match_declare_kernel,
    match_declare_kernel_cpu,
    is_cuda_target,
    is_hip_target,
    is_cpu_target,
    pythonic_expr,
    parse_function_call_args,
    parse_tma_descriptor_args,
)
import ast
import re
import logging
import textwrap
from tvm.tirx.stmt_functor import post_order_visit

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY = """
    cudaError_t result_{0} = cudaFuncSetAttribute({0}, cudaFuncAttributeMaxDynamicSharedMemorySize, {1});
    if (result_{0} != cudaSuccess) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", {1}, cudaGetErrorString(result_{0}));
        return -1;
    }}
"""

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP = """
    int device_{0} = 0;
    hipError_t dev_res_{0} = hipGetDevice(&device_{0});
    if (dev_res_{0} != hipSuccess) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to get HIP device for {0}: %s", hipGetErrorString(dev_res_{0}));
        return -1;
    }}
    int max_smem_{0} = 0;
    hipError_t attr_res_{0} = hipDeviceGetAttribute(&max_smem_{0}, hipDeviceAttributeMaxSharedMemoryPerBlock, device_{0});
    if (attr_res_{0} != hipSuccess || max_smem_{0} <= 0) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to query HIP max shared memory for {0}: %s", hipGetErrorString(attr_res_{0}));
        return -1;
    }}
    if ({1} > max_smem_{0}) {{
        snprintf(
            error_buf,
            ERROR_BUF_SIZE,
            "Requested dynamic shared memory %d exceeds device limit %d for {0}",
            {1},
            max_smem_{0}
        );
        return -1;
    }}
    return 0;
"""

PREDEF_INIT_FUNC = """
#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" TL_EXPORT const char* get_last_error() {{
    return error_buf;
}}

extern "C" TL_EXPORT int init() {{
    error_buf[0] = '\\0';
    {0}
    return 0;
}}
"""

PREDEF_HOST_FUNC = """
extern "C" TL_EXPORT int call({}) {{
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
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "Error: Failed to initialize the TMA descriptor {0}");
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
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "Error: Failed to initialize the TMA descriptor {0}");
\t\treturn -1;
\t}}
"""

KERNEL_LAUNCH_FUNC_CODE = """
\t{{
\t\tcudaLaunchConfig_t config;
\t\tcudaLaunchAttribute attribute[1];
\t\tattribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
\t\tattribute[0].val.programmaticStreamSerializationAllowed = 1;
\t\tconfig.attrs = attribute;
\t\tconfig.numAttrs = 1;
\t\tconfig.stream = stream;
\t\tconfig.gridDim = {0};
\t\tconfig.blockDim = {1};
\t\tconfig.dynamicSmemBytes = {2};
\t\tcudaLaunchKernelEx(&config, {4}, {3});
\t}}
"""

# Cluster launch code for SM90+
KERNEL_CLUSTER_LAUNCH_FUNC_CODE = """
\t{{
\t\tcudaLaunchConfig_t config;
\t\tcudaLaunchAttribute attribute[2];
\t\tattribute[0].id = cudaLaunchAttributeClusterDimension;
\t\tattribute[0].val.clusterDim = {{{5}, {6}, {7}}};
\t\tattribute[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
\t\tattribute[1].val.programmaticStreamSerializationAllowed = 1;
\t\tconfig.attrs = attribute;
\t\tconfig.numAttrs = 2;
\t\tconfig.stream = stream;
\t\tconfig.gridDim = {0};
\t\tconfig.blockDim = {1};
\t\tconfig.dynamicSmemBytes = {2};
\t\tcudaError_t cluster_attr_result = cudaFuncSetAttribute({4}, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
\t\tif (cluster_attr_result != cudaSuccess) {{
\t\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "Failed to set cluster attribute for {4}: %s", cudaGetErrorString(cluster_attr_result));
\t\t\treturn -1;
\t\t}}
\t\tcudaLaunchKernelEx(&config, {4}, {3});
\t}}
"""


class BaseWrapper(ABC):
    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError


logger = logging.getLogger(__name__)


def _require_lowered_modules(
    device_mod: IRModule | None,
    host_mod: IRModule | None,
) -> tuple[IRModule, IRModule]:
    """Require adapter metadata to come from the compiler's lowering result."""
    missing = [name for name, mod in (("device_mod", device_mod), ("host_mod", host_mod)) if mod is None]
    if missing:
        raise ValueError(f"Adapter source generation requires pre-lowered device_mod and host_mod; missing: {', '.join(missing)}.")
    return device_mod, host_mod


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
        "uint64": "uint64_t",
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
        self.pdl_sync_map: dict[str, int] | None = {}
        self.parse_source_information()
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = self.update_lib_code(source)

    def _pythonic_expr(self, expr: tvm.tirx.PrimExpr) -> str:
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
            elif isinstance(param, tvm.tirx.Var):
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
        desc_name_var_map: dict[str, tvm.tirx.Var] = {}
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = self.get_declaration(code[index:])

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
                assert self.cluster_dims[function_name] is None, "Cluster launch is not supported for cooperative groups"
                kernel_launch_code += "\tTILELANG_CHECK(cudaLaunchCooperativeKernel((void*){}, {}, {}, {}, {}, stream));\n".format(
                    function_name, grid_str, block_str, function_name + "_args", smem_str
                )
            else:
                args_list = parse_function_call_args(declaration, function_args, function_params, desc_name_map, desc_name_var_map)
                assert len(function_params) == len(args_list), (
                    f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
                )

                call_args = ", ".join(args_list)
                kernel_code = self.get_kernel_launch_code(
                    function_name, grid_str, block_str, smem_str, call_args, self.cluster_dims[function_name]
                )

                kernel_launch_code += kernel_code
                kernel_launch_code += f'\tTILELANG_CHECK_LAST_ERROR("{function_name}");\n'

            if has_l2_persistent_map:
                kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE

        init_tma_descriptor_args = self.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
        kernel_launch_code = init_tma_descriptor_args + kernel_launch_code

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        return host_func

    def get_declaration(self, declare_kernel_code: str) -> str:
        return declare_kernel_code.split(";")[0]

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

    def generate_tma_descriptor_args(self, desc_name_map: dict[str, str], desc_name_var_map: dict[str, tvm.tirx.Var]) -> str:
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

    def get_cuda_host_adapter_include(self) -> str:
        if is_cuda_target(self.target) and self.tma_descriptor_args is not None:
            return "#include <cutlass/cuda_host_adapter.hpp>\n"
        return ""

    def parse_source_information(self):
        self.device_mod, self.host_mod = _require_lowered_modules(self.device_mod, self.host_mod)
        assert len(self.device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(self.host_mod.functions) == 1, "Only support one function in host module."

        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        function_names = []
        use_cooperative_groups_map = {}
        cluster_dims_map = {}
        for g_var, func in self.device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            cluster_dims = None
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
            if "cluster_dims" in attrs:
                # Extract cluster dimensions for SM90+ cluster launch
                cluster_dims_attr = attrs["cluster_dims"]
                cluster_dims = [int(cluster_dims_attr[i]) for i in range(len(cluster_dims_attr))]

            if "has_cuda_pdl_sync" in attrs:
                self.pdl_sync_map[function_name] = 0

            # Map the extracted configurations to each function
            block_info_map[function_name] = block_info
            grid_info_map[function_name] = grid_info
            dynamic_smem_buf_map[function_name] = dynamic_smem_buf
            use_cooperative_groups_map[function_name] = use_cooperative_groups
            cluster_dims_map[function_name] = cluster_dims
            function_names.append(function_name)

        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map
        self.use_cooperative_groups = use_cooperative_groups_map
        self.cluster_dims = cluster_dims_map

        function_names_index = {}
        for g_var, func in self.host_mod.functions.items():
            function_name = g_var.name_hint
            if "tma_descriptor_args" in func.attrs:
                self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
            if "l2_persistent_map" in func.attrs:
                self.l2_persistent_map[function_name] = func.attrs["l2_persistent_map"]

            host_code = str(func)
            for function_name in function_names:
                try:
                    index = host_code.index(f'T.call_packed("{function_name}"')
                except ValueError:
                    index = host_code.index(f'value="{function_name}"')
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
                    if isinstance(dim, tvm.tirx.Var):
                        unique_push_back(dim.name, str(dim.dtype))

        # Note: In buffer definitions, any dynamic symbols appearing in strides are listed after those in the shape.
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for stride in buffer.strides:
                    if isinstance(stride, tvm.tirx.Var):
                        unique_push_back(stride.name, str(stride.dtype))

        return list(dynamic_symbolic_set.items())

    def get_kernel_launch_code(self, function_name, grid_str, block_str, smem_str, call_args, cluster_dims):
        if cluster_dims is None:
            return KERNEL_LAUNCH_FUNC_CODE.format(grid_str, block_str, smem_str, call_args, function_name)
        else:
            return KERNEL_CLUSTER_LAUNCH_FUNC_CODE.format(grid_str, block_str, smem_str, call_args, function_name, *cluster_dims)

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
                if isinstance(node, tvm.tirx.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tirx.tvm_call_packed")):
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
                "cluster_dims": self.cluster_dims.get(function_name, None),
            }

        # Create the host function wrapper for the CUDA kernel
        host_func = self.create_dispatch_func(code, function_informations)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + self.get_cuda_host_adapter_include() + init_func + host_func
        return lib_code

    def get_stream_type(self) -> dict[str, str] | None:
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
        "float8_e5m2fnuz": "fp8_e5_t",
        "float8_e4m3fnuz": "fp8_e4_t",
        "e4m3fnuz_float8": "fp8_e4_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "uint64": "uint64_t",
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

    def get_declaration(self, declare_kernel_code: str) -> str:
        # HIP code dont have function declaration, so we use '{\n' to split
        # __global__ void __launch_bounds__(128) kernel_kernel(float* __restrict__ A) {\n
        return declare_kernel_code.split("{")[0]

    def get_kernel_launch_code(self, function_name, grid_str, block_str, smem_str, call_args, cluster_dims):
        # HIP does not support cudaLaunchKernelEx; use <<<>>> syntax (same as pre-cluster-launch behavior)
        return f"\t{function_name}<<<{grid_str}, {block_str}, {smem_str}, stream>>>({call_args});\n"

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
            elif isinstance(param, tvm.tirx.Var):
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
        self.device_mod, self.host_mod = _require_lowered_modules(self.device_mod, self.host_mod)
        assert len(self.device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(self.host_mod.functions) == 1, "Only support one function in host module."

        function_names = []
        for g_var, _ in self.device_mod.functions.items():
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
                    if isinstance(dim, tvm.tirx.Var) and (dim.name not in dynamic_symbolic_set):
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


class TLAscendSourceWrapper(TLCUDASourceWrapper):
    """Wrapper for Ascend backend kernel source.

    Ascend kernels use ``__global__ __vector__`` qualifiers and the triple-
    chevron launch syntax ``<<<grid, smem, stream>>>`` (no block dim).
    This wrapper reuses the CUDA wrapper infrastructure but overrides the
    launch code generation and init/stream helpers for ACL runtime.
    """

    _TYPE_MAP = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e4m3fn": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float8_e8m0fnu": "fp8_e8_t",
        "float4_e2m1fn": "float4_e2m1x2_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
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

    def get_declaration(self, declare_kernel_code: str) -> str:
        # Ascend kernel declarations end with ')' before '{', no forward decl ';'
        return declare_kernel_code.split("{")[0]

    def create_dispatch_func(self, code, function_informations):
        """Generate the host ``call()`` function with Ascend launch semantics.

        Ascend uses ``<<<grid, smem, stream>>>`` (three arguments, no block
        dim) and ACL error checking instead of CUDA equivalents.
        """
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append(
                    {
                        "name": buffer.data.name,
                        "type": self._lookup_type(buffer.dtype) + "* __restrict__",
                    }
                )
            elif isinstance(param, tvm.tirx.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")
        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": self._lookup_type(dyn_sym_dtype)})

        stream_type = self.get_stream_type()
        if stream_type is not None:
            function_args.append(stream_type)

        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        kernel_launch_code = """"""
        for function_name, function_info in function_informations.items():
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            # Ascend kernel signature: __global__ __vector__ <ret> name(...)
            # match_declare_kernel expects __global__ void, so find by name directly.
            func_decl_pos = code.index(function_name + "(")
            declaration = self.get_declaration(code[func_decl_pos:])
            brace_pos = code.index("{", func_decl_pos)  # noqa: F841

            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf

            args_list = parse_function_call_args(declaration, function_args, function_params, {}, {})
            assert len(function_params) == len(args_list), (
                f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
            )
            call_args = ", ".join(args_list)
            # Ascend triple-chevron: <<<grid, smem, stream>>> (no block dim)
            # Ascend bisheng expects grid as a single unsigned int, not dim3
            grid_dim = f"({self._pythonic_expr(grid_info[0])} * {self._pythonic_expr(grid_info[1])} * {self._pythonic_expr(grid_info[2])})"
            stream_arg = "stream" if stream_type else "nullptr"
            kernel_launch_code += f"\t{function_name}<<<{grid_dim}, {smem_str}, {stream_arg}>>>({call_args});\n"
            kernel_launch_code += f'\tTILELANG_CHECK_LAST_ERROR("{function_name}");\n'

        host_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        return host_func

    def get_init_func(self):
        # Ascend does not need cudaFuncSetAttribute for dynamic shared memory
        init_funcs = PREDEF_INIT_FUNC.format("")
        return init_funcs

    def update_lib_code(self, code: str):
        # Prepend Ascend common header so TILELANG_CHECK macros and dim3 are available
        # for both kernel code and wrapper code
        ascend_header = "#include <tl_templates/ascend/common.h>\n"
        lib_code = super().update_lib_code(code)
        # The parent class combines source + init_func + host_func.
        # We need the header at the very beginning, so prepend it to the final result.
        if ascend_header not in lib_code:
            lib_code = ascend_header + lib_code
        return lib_code

    def get_stream_type(self) -> dict[str, str]:
        # aclrtStream is typedef void* (defined in acl_base_rt.h:46)
        # Use void* directly to avoid depending on ACL headers in the host wrapper.
        # The Cython caller will pass the raw handle obtained from
        # torch.npu.current_stream().npu_stream.
        return {"name": "stream=nullptr", "type": "void*"}


@dataclass(frozen=True)
class _PTOKernelDescriptor:
    name: str
    device_func: tvm.tirx.PrimFunc
    ptodsl_arg_names: tuple[str, ...]
    prototype_types: tuple[str, ...]
    launch_param_tags: tuple[str, ...] | None


@dataclass(frozen=True)
class _PTOKernelCallSite:
    kernel: _PTOKernelDescriptor
    function_args: tuple[tvm.tirx.PrimExpr, ...]
    launch_args: tuple[tvm.tirx.PrimExpr, ...]


@dataclass(frozen=True)
class _PTOHostKernelCall:
    name: str
    args: tuple[tvm.tirx.PrimExpr, ...]


@tvm.tirx.functor.visitor
class _PTOHostCallCollector(tvm.tirx.functor.PyStmtExprVisitor):
    """Collect direct PTO kernel launches while preserving host statement order."""

    def __init__(self, device_func_by_name: dict[str, tvm.tirx.PrimFunc]):
        super().__init__()
        self.device_func_by_name = device_func_by_name
        self.kernel_calls: list[_PTOHostKernelCall] = []
        self.control_flow_depth = 0

    @staticmethod
    def _packed_call_name(arg: Any) -> str | None:
        if isinstance(arg, str):
            return arg
        if isinstance(arg, tvm.tirx.StringImm):
            return arg.value
        return None

    def _kernel_from_call(self, call: tvm.tirx.Call) -> str | None:
        if not call.op.same_as(tvm.ir.Op.get("tirx.tvm_call_packed")) or not call.args:
            return None
        name = self._packed_call_name(call.args[0])
        if name is None or name not in self.device_func_by_name:
            return None
        return name

    def visit_evaluate_(self, op: tvm.tirx.Evaluate) -> None:
        value = op.value
        if isinstance(value, tvm.tirx.Call):
            kernel_name = self._kernel_from_call(value)
            if kernel_name is not None:
                if self.control_flow_depth:
                    raise RuntimeError("PTO JIT multi-kernel host launcher does not yet support kernel calls under host control flow.")
                self.kernel_calls.append(_PTOHostKernelCall(kernel_name, tuple(value.args[1:])))
                return
        self.visit_expr(value)

    def visit_call_(self, op: tvm.tirx.Call) -> None:
        kernel_name = self._kernel_from_call(op)
        if kernel_name is not None:
            raise RuntimeError(f"PTO kernel call `{kernel_name}` must be a direct host Evaluate statement.")
        for arg in op.args:
            self.visit_expr(arg)

    def visit_seq_stmt_(self, op: tvm.tirx.SeqStmt) -> None:
        for stmt in op.seq:
            self.visit_stmt(stmt)

    def _visit_control_flow_body(self, body) -> None:
        self.control_flow_depth += 1
        try:
            self.visit_stmt(body)
        finally:
            self.control_flow_depth -= 1

    def visit_if_then_else_(self, op: tvm.tirx.IfThenElse) -> None:
        self.visit_expr(op.condition)
        self._visit_control_flow_body(op.then_case)
        if op.else_case is not None:
            self._visit_control_flow_body(op.else_case)

    def visit_for_(self, op: tvm.tirx.For) -> None:
        self.visit_expr(op.min)
        self.visit_expr(op.extent)
        self._visit_control_flow_body(op.body)

    def visit_while_(self, op: tvm.tirx.While) -> None:
        self.visit_expr(op.condition)
        self._visit_control_flow_body(op.body)


class TLPTOSourceWrapper:
    """Wrapper for PTO JIT source.

    PTO codegen emits PTODSL source. This wrapper generates the host launch
    stub directly from PTODSL/TIR metadata and keeps the PTODSL source for
    device compilation in libgen.
    """

    _TYPE_MAP = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "float8_e4m3_t",
        "float8_e4m3fn": "float8_e4m3_t",
        "float8_e5m2": "float8_e5m2_t",
        "float8_e8m0fnu": "float8_e8m0_t",
        "float4_e2m1fn": "float4_e2m1x2_t",
        "float4_e2m1fnx2": "float4_e2m1x2_t",
        "float64": "double",
        "int64": "int64_t",
        "uint64": "uint64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
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
        # Preserve the generated PTODSL module for signature parsing and device compilation.
        self.pto_kernel_source = source.strip()
        # Select the scheduled entry that defines the exported host-call ABI.
        prim_func = self._select_scheduled_entry_func(scheduled_ir_module)
        # Build a lightweight symbol table for all lowered device functions.
        device_func_by_name = self._collect_device_functions(device_mod)
        # Select the lowered host entry that contains the actual launch sequence.
        host_func = self._select_host_entry_func(host_mod)
        # Collect kernel calls from the host entry in execution order.
        kernel_calls = self._collect_host_kernel_calls(host_func, device_func_by_name)
        # Derive the unique kernel compilation list while preserving first-use order.
        self.pto_kernel_names = self._ordered_unique_kernel_names(kernel_calls)
        # Parse signatures only for kernels referenced by the host entry.
        ptodsl_signatures = self._parse_ptodsl_kernel_signatures(self.pto_kernel_names)
        # Validate the referenced kernels and materialize their ABI and launch metadata.
        kernel_by_name = self._build_kernel_descriptors(self.pto_kernel_names, device_func_by_name, ptodsl_signatures)
        # Split each packed call into device arguments and launch arguments.
        kernel_call_sites = self._resolve_host_kernel_call_sites(kernel_calls, kernel_by_name)
        # Generate one exported host function that launches all call sites in order.
        self.lib_code = self._generate_host_source(prim_func, self.pto_kernel_names, kernel_call_sites)

    @staticmethod
    def _select_scheduled_entry_func(mod: IRModule) -> tvm.tirx.PrimFunc:
        if len(mod.get_global_vars()) == 1:
            return mod[mod.get_global_vars()[0]]
        if "main" in mod:
            return mod["main"]
        for _, function in mod.functions.items():
            attr = function.attrs
            if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                return function
        raise ValueError("Cannot find primary function in the module.")

    def _pythonic_expr(self, expr: tvm.tirx.PrimExpr) -> str:
        return pythonic_expr(
            expr,
            self._TYPE_MAP,
            floor_div_op="/",
            expression_style="cxx",
        )

    def _parse_ptodsl_kernel_signatures(self, kernel_names: list[str]) -> dict[str, tuple[str, ...]]:
        try:
            module = ast.parse(self.pto_kernel_source)
        except SyntaxError as err:
            raise RuntimeError("Failed to parse PTODSL source for PTO kernels.") from err

        required_names = set(kernel_names)
        definitions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in required_names}
        signatures: dict[str, tuple[str, ...]] = {}
        for kernel_name in kernel_names:
            node = definitions.get(kernel_name)
            if node is None:
                raise RuntimeError(f"Cannot find PTODSL function definition for PTO kernel `{kernel_name}`.")
            if node.args.posonlyargs or node.args.vararg or node.args.kwonlyargs or node.args.kwarg:
                raise RuntimeError(f"PTO kernel `{node.name}` must use positional arguments only.")
            arg_names = tuple(arg.arg for arg in node.args.args)
            if len(arg_names) != len(set(arg_names)):
                raise RuntimeError(f"PTO kernel `{node.name}` has duplicate argument names: {list(arg_names)}")
            signatures[node.name] = arg_names
        return signatures

    def _device_param_prototype_type(self, kernel_name: str, param: tvm.tirx.Var) -> str:
        annotation = param.type_annotation
        if isinstance(annotation, tvm.ir.PointerType):
            storage_scope = str(annotation.storage_scope)
            if storage_scope != "global":
                raise RuntimeError(
                    f"PTO kernel `{kernel_name}` parameter `{param.name}` uses unsupported pointer storage scope `{storage_scope}`."
                )
            element_type = annotation.element_type
            if not isinstance(element_type, tvm.ir.PrimType):
                raise RuntimeError(f"PTO kernel `{kernel_name}` parameter `{param.name}` has unsupported pointer type `{annotation}`.")
            return self._gm_cast_type(element_type.dtype)
        if str(param.dtype) == "handle":
            raise RuntimeError(f"PTO kernel `{kernel_name}` parameter `{param.name}` has an unsupported opaque handle type.")
        return self._lookup_type(param.dtype)

    @staticmethod
    def _collect_device_functions(device_mod: IRModule | None) -> dict[str, tvm.tirx.PrimFunc]:
        if device_mod is None:
            raise RuntimeError("PTO wrapper requires a device module.")

        device_funcs: dict[str, tvm.tirx.PrimFunc] = {}
        for gvar, func in device_mod.functions.items():
            if not isinstance(func, tvm.tirx.PrimFunc):
                continue
            name = str(func.attrs["global_symbol"]) if "global_symbol" in func.attrs else gvar.name_hint
            if name in device_funcs:
                raise RuntimeError(f"Duplicate PTO device kernel symbol: `{name}`.")
            device_funcs[name] = func

        if not device_funcs:
            raise RuntimeError("PTO wrapper requires at least one device kernel.")
        return device_funcs

    def _build_kernel_descriptors(
        self,
        kernel_names: list[str],
        device_func_by_name: dict[str, tvm.tirx.PrimFunc],
        ptodsl_signatures: dict[str, tuple[str, ...]],
    ) -> dict[str, _PTOKernelDescriptor]:
        kernels: dict[str, _PTOKernelDescriptor] = {}
        for name in kernel_names:
            func = device_func_by_name[name]
            ptodsl_arg_names = ptodsl_signatures[name]
            if len(ptodsl_arg_names) != len(func.params):
                raise RuntimeError(
                    f"PTO kernel `{name}` argument count mismatch: PTODSL={len(ptodsl_arg_names)}, device={len(func.params)}."
                )
            prototype_types = tuple(self._device_param_prototype_type(name, param) for param in func.params)
            launch_param_tags = (
                tuple(str(tag) for tag in func.attrs["tirx.kernel_launch_params"]) if "tirx.kernel_launch_params" in func.attrs else None
            )
            if launch_param_tags is not None and len(launch_param_tags) != len(set(launch_param_tags)):
                raise RuntimeError(f"PTO kernel `{name}` has duplicate launch parameter tags: {list(launch_param_tags)}")
            kernels[name] = _PTOKernelDescriptor(
                name=name,
                device_func=func,
                ptodsl_arg_names=ptodsl_arg_names,
                prototype_types=prototype_types,
                launch_param_tags=launch_param_tags,
            )
        return kernels

    @staticmethod
    def _select_host_entry_func(host_mod: IRModule | None) -> tvm.tirx.PrimFunc:
        if host_mod is None:
            raise RuntimeError("PTO wrapper requires a host module to determine kernel launch order.")
        functions = [(gvar, func) for gvar, func in host_mod.functions.items() if isinstance(func, tvm.tirx.PrimFunc)]
        if len(functions) == 1:
            return functions[0][1]

        def select_unique(candidates, description):
            if len(candidates) > 1:
                names = [gvar.name_hint for gvar, _ in candidates]
                raise RuntimeError(f"Found multiple PTO host {description} functions: {names}")
            return candidates[0][1] if candidates else None

        entry = select_unique(
            [(gvar, func) for gvar, func in functions if "tirx.is_entry_func" in func.attrs and bool(func.attrs["tirx.is_entry_func"])],
            "entry",
        )
        if entry is not None:
            return entry
        entry = select_unique([(gvar, func) for gvar, func in functions if gvar.name_hint == "main"], "main")
        if entry is not None:
            return entry
        entry = select_unique(
            [
                (gvar, func)
                for gvar, func in functions
                if "global_symbol" in func.attrs and str(func.attrs["global_symbol"]) == "__tvm_ffi_main"
            ],
            "global entry",
        )
        if entry is not None:
            return entry
        raise RuntimeError("Cannot find PTO host entry function.")

    @staticmethod
    def _collect_host_kernel_calls(
        host_func: tvm.tirx.PrimFunc,
        device_func_by_name: dict[str, tvm.tirx.PrimFunc],
    ) -> list[_PTOHostKernelCall]:
        collector = _PTOHostCallCollector(device_func_by_name)
        collector.visit_stmt(host_func.body)
        if not collector.kernel_calls:
            raise RuntimeError("No PTO kernel call sites found in host entry function.")
        return collector.kernel_calls

    @staticmethod
    def _ordered_unique_kernel_names(kernel_calls: list[_PTOHostKernelCall]) -> list[str]:
        return list(dict.fromkeys(kernel_call.name for kernel_call in kernel_calls))

    @staticmethod
    def _resolve_host_kernel_call_sites(
        kernel_calls: list[_PTOHostKernelCall],
        kernel_by_name: dict[str, _PTOKernelDescriptor],
    ) -> list[_PTOKernelCallSite]:
        call_sites: list[_PTOKernelCallSite] = []
        for kernel_call in kernel_calls:
            kernel = kernel_by_name[kernel_call.name]
            if kernel.launch_param_tags is None:
                raise RuntimeError(f"PTO kernel `{kernel.name}` is missing `tirx.kernel_launch_params` from LowerDeviceKernelLaunch.")
            param_count = len(kernel.device_func.params)
            expected_count = param_count + len(kernel.launch_param_tags)
            if len(kernel_call.args) != expected_count:
                raise RuntimeError(
                    f"PTO host call `{kernel.name}` argument count mismatch: expected {expected_count}, got {len(kernel_call.args)}."
                )
            call_sites.append(
                _PTOKernelCallSite(
                    kernel=kernel,
                    function_args=kernel_call.args[:param_count],
                    launch_args=kernel_call.args[param_count:],
                )
            )
        return call_sites

    def _lookup_type(self, dtype: str | Any) -> str:
        key = dtype if isinstance(dtype, str) else str(dtype)
        result = self._TYPE_MAP.get(key)
        if result is None:
            raise RuntimeError(f"Unsupported PTO argument dtype: {dtype}")
        return result

    def _gm_cast_type(self, dtype: str | Any) -> str:
        return f"__gm__ {self._lookup_type(dtype)} *"

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
                    if isinstance(dim, tvm.tirx.Var):
                        unique_push_back(dim.name, str(dim.dtype))

        # Note: In buffer definitions, any dynamic symbols appearing in strides are listed after those in the shape.
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for stride in buffer.strides:
                    if isinstance(stride, tvm.tirx.Var):
                        unique_push_back(stride.name, str(stride.dtype))

        return list(dynamic_symbolic_set.items())

    def _host_argument_infos(self, func) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(func)
        host_args = []
        arg_by_name = {}

        def add_alias(alias: str, info: dict[str, Any]):
            if alias in arg_by_name and arg_by_name[alias]["name"] != info["name"]:
                raise RuntimeError(f"Duplicate PTO host argument name or alias: {alias}")
            arg_by_name[alias] = info

        for param in func.params:
            if param in func.buffer_map:
                buffer = func.buffer_map[param]
                name = buffer.data.name
                info = {
                    "name": name,
                    "host_type": "void *",
                    "kind": "pointer",
                }
                host_args.append(info)
                add_alias(name, info)
                add_alias(param.name, info)
                add_alias(buffer.name, info)
            elif isinstance(param, tvm.tirx.Var):
                name = param.name
                c_type = self._lookup_type(param.dtype)
                info = {
                    "name": name,
                    "host_type": c_type,
                    "kind": "scalar",
                }
                host_args.append(info)
                add_alias(name, info)
            else:
                raise RuntimeError(f"Unsupported PTO kernel parameter: {param}")

        # Add dynamic symbols as integer arguments
        existing_names = {arg["name"] for arg in host_args}
        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            if dyn_sym in existing_names:
                continue
            c_type = self._lookup_type(dyn_sym_dtype)
            info = {
                "name": dyn_sym,
                "host_type": c_type,
                "kind": "scalar",
            }
            host_args.append(info)
            add_alias(dyn_sym, info)

        if len({arg["name"] for arg in host_args}) != len(host_args):
            names = [arg["name"] for arg in host_args]
            raise RuntimeError(f"Duplicate PTO host argument names: {names}")

        return host_args, arg_by_name

    @staticmethod
    def _is_pointer_param(param: tvm.tirx.Var) -> bool:
        return isinstance(param.type_annotation, tvm.ir.PointerType)

    def _render_scalar_expr(
        self,
        expr: tvm.tirx.PrimExpr | int,
        host_arg_by_name: dict[str, dict[str, Any]],
        context: str,
    ) -> str:
        if not isinstance(expr, tvm.tirx.PrimExpr):
            return str(expr)

        allowed_types = (
            tvm.tirx.Var,
            tvm.tirx.IntImm,
            tvm.tirx.FloatImm,
            tvm.tirx.Cast,
            tvm.tirx.Mul,
            tvm.tirx.FloorDiv,
            tvm.tirx.Add,
            tvm.tirx.Sub,
            tvm.tirx.FloorMod,
            tvm.tirx.Min,
            tvm.tirx.Max,
            tvm.tirx.LT,
            tvm.tirx.LE,
            tvm.tirx.GT,
            tvm.tirx.GE,
            tvm.tirx.EQ,
            tvm.tirx.NE,
            tvm.tirx.And,
            tvm.tirx.Or,
        )
        substitutions = {}
        unsupported = []

        def visitor(node):
            if isinstance(node, tvm.tirx.Var):
                info = host_arg_by_name.get(node.name)
                if info is None:
                    raise RuntimeError(f"{context} references unknown host variable `{node.name}`.")
                if info["kind"] != "scalar":
                    raise RuntimeError(f"{context} uses pointer host variable `{node.name}` as a scalar expression.")
                substitutions[node] = tvm.tirx.Var(info["name"], node.dtype)
            elif isinstance(node, tvm.tirx.PrimExpr) and not isinstance(node, allowed_types):
                unsupported.append(type(node).__name__)

        post_order_visit(expr, visitor)
        if unsupported:
            raise RuntimeError(f"{context} contains unsupported expression nodes: {sorted(set(unsupported))}.")
        if substitutions:
            expr = tvm.tirx.stmt_functor.substitute(expr, substitutions)
        return self._pythonic_expr(expr)

    def _render_call_arg(
        self,
        call_site: _PTOKernelCallSite,
        index: int,
        host_arg_by_name: dict[str, dict[str, Any]],
    ) -> str:
        kernel = call_site.kernel
        expr = call_site.function_args[index]
        param = kernel.device_func.params[index]
        prototype_type = kernel.prototype_types[index]
        context = f"PTO kernel `{kernel.name}` argument {index} (`{kernel.ptodsl_arg_names[index]}`)"
        if self._is_pointer_param(param):
            if not isinstance(expr, tvm.tirx.Var):
                raise RuntimeError(f"{context} requires a direct host pointer variable, got `{expr}`.")
            info = host_arg_by_name.get(expr.name)
            if info is None or info["kind"] != "pointer":
                raise RuntimeError(f"{context} cannot map host pointer variable `{expr.name}` to the exported call ABI.")
            return f"({prototype_type}){info['name']}"

        rendered = self._render_scalar_expr(expr, host_arg_by_name, context)
        return f"({prototype_type})({rendered})"

    def _launch_metadata(
        self,
        call_site: _PTOKernelCallSite,
        host_arg_by_name: dict[str, dict[str, Any]],
    ) -> tuple[str, str]:
        kernel = call_site.kernel
        if kernel.launch_param_tags is None:
            raise RuntimeError(f"PTO kernel `{kernel.name}` call site has no launch parameter metadata.")
        supported_tags = {
            "blockIdx.x",
            "blockIdx.y",
            "blockIdx.z",
            "tirx.use_dyn_shared_memory",
        }
        unsupported_tags = [tag for tag in kernel.launch_param_tags if tag not in supported_tags]
        if unsupported_tags:
            raise RuntimeError(
                f"PTO kernel `{kernel.name}` has unsupported launch parameter tags "
                f"{unsupported_tags}; all tags: {list(kernel.launch_param_tags)}."
            )
        launch_values = dict(zip(kernel.launch_param_tags, call_site.launch_args))
        grid_exprs = []
        for tag in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
            expr = launch_values.get(tag, 1)
            grid_exprs.append(self._render_scalar_expr(expr, host_arg_by_name, f"PTO kernel `{kernel.name}` grid `{tag}`"))

        dynamic_smem = launch_values.get("tirx.use_dyn_shared_memory", 0)
        dynamic_smem_str = self._render_scalar_expr(
            dynamic_smem,
            host_arg_by_name,
            f"PTO kernel `{kernel.name}` dynamic shared memory",
        )
        return f"({grid_exprs[0]} * {grid_exprs[1]} * {grid_exprs[2]})", dynamic_smem_str

    def _generate_host_source(
        self,
        func: tvm.tirx.PrimFunc,
        kernel_names: list[str],
        call_sites: list[_PTOKernelCallSite],
    ) -> str:
        host_args, host_arg_by_name = self._host_argument_infos(func)
        launch_params = [f"{arg['host_type']} {arg['name']}" for arg in host_args]
        launch_params.append("void *stream")

        prototypes = []
        descriptor_by_name = {call_site.kernel.name: call_site.kernel for call_site in call_sites}
        for kernel_name in kernel_names:
            kernel = descriptor_by_name[kernel_name]
            prototypes.append(f'extern "C" __global__ AICORE void {kernel.name}({", ".join(kernel.prototype_types)});')

        launches = []
        for call_site in call_sites:
            grid_dim, dynamic_smem = self._launch_metadata(call_site, host_arg_by_name)
            call_args = [self._render_call_arg(call_site, index, host_arg_by_name) for index in range(len(call_site.function_args))]
            launches.append(f"  {call_site.kernel.name}<<<{grid_dim}, {dynamic_smem}, stream>>>({', '.join(call_args)});")

        prototype_source = "#ifndef AICORE\n#define AICORE [aicore]\n#endif\n" + "\n".join(prototypes)
        launch_stub = f'extern "C" TL_EXPORT int call({", ".join(launch_params)}) {{\n' + "\n".join(launches) + "\n  return 0;\n}\n"
        return "\n\n".join(
            ["#include <algorithm>", PREDEF_INIT_FUNC.format(""), prototype_source, launch_stub]
        )


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
    pto_kernel_source: str | None = None
    pto_kernel_names: list[str] | None = None

    def __init__(self, target: Target):
        super().__init__()
        self.scheduled_ir_module = None
        self.pass_configs = None
        self.target = target
        self.lib = None
        self.pto_kernel_source = None
        self.pto_kernel_names = None

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
        elif is_pto_target(self.target):
            wrapper_class = TLPTOSourceWrapper
        elif is_ascend_target(self.target):
            wrapper_class = TLAscendSourceWrapper
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
        self.pto_kernel_source = getattr(wrapper, "pto_kernel_source", None)
        self.pto_kernel_names = getattr(wrapper, "pto_kernel_names", None)
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
