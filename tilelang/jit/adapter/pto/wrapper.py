"""Host launch wrappers and ABI validation for PTODSL kernels."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from tilelang import tvm
from tvm import IRModule
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from tilelang.jit.adapter.utils import pythonic_expr
from tilelang.jit.adapter.wrapper import PREDEF_INIT_FUNC, TLWrapper


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
        return "\n\n".join(["#include <algorithm>", PREDEF_INIT_FUNC.format(""), prototype_source, launch_stub])


class TLPTOWrapper(TLWrapper):
    def wrap(self, source: str):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        self.source_wrapper = TLPTOSourceWrapper(
            scheduled_ir_module=self.scheduled_ir_module,
            source=source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return self.source_wrapper.lib_code
