"""PTO compilation metadata and host launches from the native Function graph."""

from __future__ import annotations

import ast

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT, SourceHostProgram
from tilelang.jit.adapter.wrapper import TLWrapper, _require_lowered_modules
from tilelang.utils.language import retrieve_func_from_module


class TLPTOSourceWrapper:
    """Keep the PTODSL device module and emit its actual typed host call graph.

    Python source parsing checks the compiler's device signature for consistency.
    Call arguments, control flow, shape variables and launch metadata all come
    from native IR, including after independent JSON serialization.
    """

    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        self.mod = scheduled_ir_module
        self.target = target
        self.pass_configs = pass_configs
        self.device_mod, self.host_mod = _require_lowered_modules(device_mod, host_mod)
        self.prim_func = retrieve_func_from_module(scheduled_ir_module)
        self.program = SourceHostProgram(self.prim_func, self.host_mod)
        self.pto_kernel_source = source.strip()
        self.pto_kernel_names = self._referenced_kernels()
        self._validate_device_signatures()
        if target.kind.name != "ascend" or "pto" not in target.keys:
            raise ValueError("PTO source wrapper requires an Ascend target with the pto key")
        self.lib_code = NATIVE_SOURCE_INIT + self.program.emit_device(self.device_mod, target)

    def _referenced_kernels(self):
        names, seen_functions = [], set()

        def visit_function(global_var):
            if global_var in seen_functions:
                return
            seen_functions.add(global_var)
            function = self.program.functions[global_var]

            def visit_call(call):
                if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_ffi_kernel":
                    if not call.args or not isinstance(call.args[0], ir.StringImm):
                        raise ValueError("PTO kernel launch requires a literal device symbol")
                    name = call.args[0].value
                    if name not in names:
                        names.append(name)
                elif isinstance(call.op, ir.GlobalVar):
                    if call.op not in self.program.functions:
                        raise ValueError("PTO native host graph is missing a called helper")
                    visit_function(call.op)

            tvm_ffi.structural_walk(function.body, (ir.Call, visit_call), order="pre")

        visit_function(self.program.entry_global)
        if not names:
            raise RuntimeError("No PTO kernel call sites found in the native host graph")
        return names

    def _validate_device_signatures(self):
        try:
            module = ast.parse(self.pto_kernel_source)
        except SyntaxError as error:
            raise RuntimeError("Failed to parse PTODSL source for PTO kernels") from error
        definitions = {}
        for statement in module.body:
            if isinstance(statement, ast.FunctionDef) and statement.name in self.pto_kernel_names:
                if statement.name in definitions:
                    raise RuntimeError(f"Duplicate PTODSL kernel definition: {statement.name}")
                definitions[statement.name] = statement
        device_functions = {}
        for global_var, function in self.device_mod.functions.items():
            if not isinstance(function, tirx.Function):
                raise TypeError("PTO device module requires native Function objects")
            name = function.attrs.get("global_symbol", global_var.name_hint)
            if name in device_functions:
                raise RuntimeError(f"Duplicate PTO device kernel symbol: {name}")
            device_functions[name] = function
        for name in self.pto_kernel_names:
            if name not in device_functions:
                raise RuntimeError(f"Missing PTO device kernel: {name}")
            node = definitions.get(name)
            if node is None:
                raise RuntimeError(f"Cannot find PTODSL function definition for PTO kernel `{name}`")
            args = node.args
            if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
                raise RuntimeError(f"PTO kernel `{name}` must use positional arguments only")
            names = [argument.arg for argument in args.args]
            if len(names) != len(set(names)):
                raise RuntimeError(f"PTO kernel `{name}` has duplicate argument names")
            if len(names) != len(device_functions[name].params):
                raise RuntimeError(f"PTO kernel `{name}` PTODSL signature disagrees with its native Function")


class TLPTOWrapper(TLWrapper):
    def wrap(self, source: str):
        if self.scheduled_ir_module is None:
            raise ValueError("Please assign optimized module first")
        self.source_wrapper = TLPTOSourceWrapper(
            self.scheduled_ir_module,
            source,
            self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return self.source_wrapper.lib_code
