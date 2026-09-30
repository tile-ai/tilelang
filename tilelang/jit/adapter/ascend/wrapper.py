"""Host launch wrappers for AscendC kernels."""

from __future__ import annotations

from typing import Any

from tilelang import tvm
from tvm import IRModule
from tvm.target import Target

from tilelang.jit.adapter.utils import parse_function_call_args
from tilelang.jit.adapter.wrapper import PREDEF_HOST_FUNC, PREDEF_INIT_FUNC, TLCUDASourceWrapper, TLWrapper


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


class TLAscendWrapper(TLWrapper):
    source_wrapper_class = TLAscendSourceWrapper
