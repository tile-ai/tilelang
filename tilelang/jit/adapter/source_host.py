"""Native host graphs for C, CUDA, HIP, and Ascend source adapters.

The compiler retains this graph before MakePackedAPI. Source adapters project
its tensor parameters to pointers while preserving native statements, variable
identity, and call operands. TileLang's C emitter owns local names and control
flow; no generated source or printed IR is parsed to recover an ABI.
"""

from __future__ import annotations

from collections.abc import Callable

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from .abi import SourceABIPlan


NATIVE_SOURCE_INIT = r"""
#include <tvm/ffi/c_api.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define TL_SOURCE_EXPORT __declspec(dllexport)
#else
#define TL_SOURCE_EXPORT
#endif
static thread_local char tl_source_error[1024];
extern "C" int32_t __tl_source_begin_call() {
  TVMFFIObjectHandle error = nullptr;
  TVMFFIErrorMoveFromRaised(&error);
  if (error != nullptr) TVMFFIObjectDecRef(error);
  tl_source_error[0] = '\0';
  return 0;
}
extern "C" TL_SOURCE_EXPORT const char* get_last_error() {
  TVMFFIObjectHandle error = nullptr;
  TVMFFIErrorMoveFromRaised(&error);
  if (error != nullptr) {
    const auto* cell = TVMFFIErrorGetCellPtr(error);
    size_t length = cell->message.size;
    if (length >= sizeof(tl_source_error)) length = sizeof(tl_source_error) - 1;
    snprintf(tl_source_error, sizeof(tl_source_error), "%.*s", (int)length, cell->message.data);
    TVMFFIObjectDecRef(error);
  }
  return tl_source_error;
}
extern "C" TL_SOURCE_EXPORT int init() { return __tl_source_begin_call(); }
"""


def source_host_functions(host_module: ir.IRModule) -> dict[ir.GlobalVar, tirx.Function]:
    attrs = host_module.attrs
    if not attrs or int(attrs.get("tl.source_host_schema", 0)) != 2:
        raise ValueError("Source adapters require the native host graph retained by MakePackedAPI")
    graph = attrs.get("tl.source_host_module")
    if not isinstance(graph, ir.IRModule):
        raise ValueError("Malformed native source-host module")
    functions = dict(graph.functions.items())
    if not functions or any(not isinstance(value, tirx.Function) for value in functions.values()):
        raise ValueError("Malformed native source-host function table")
    return functions


class SourceHostProgram:
    """A raw pointer/scalar entry over a compiler-produced native host graph."""

    def __init__(self, public_function: tirx.Function, host_module: ir.IRModule):
        self.functions = source_host_functions(host_module)
        symbol = public_function.attrs.get("global_symbol")
        entries = [(key, value) for key, value in self.functions.items() if value.attrs.get("global_symbol") == symbol]
        if len(entries) != 1:
            raise ValueError(f"Expected one native source host entry for {symbol!r}")
        self.entry_global, self.entry = entries[0]
        self.public_abi = SourceABIPlan(public_function)
        self.abi = SourceABIPlan(self.entry)
        self.abi.validate_parameter_metadata(self.public_abi.parameters)
        self._validate_signature_graph()

    @staticmethod
    def _signature_graph(plan):
        def value(expression):
            # A layout canonicalizer may widen a constant unit stride. Its
            # represented integer is the ABI contract; variable types and
            # nonconstant arithmetic remain part of the comparison below.
            return int(expression.value) if isinstance(expression, ir.IntImm) else expression

        return tvm_ffi.convert(
            [
                [
                    [[value(item) for item in tensor.shape], [value(item) for item in tensor.strides], value(tensor.element_offset)]
                    for tensor in plan.tensors.values()
                ],
                [parameter for parameter in plan.function.params if isinstance(parameter.ty, ir.PrimType)],
                plan.implicit_symbols,
            ]
        )

    def _validate_signature_graph(self):
        # Include scalar identities as well as dimensions. A scalar appearing
        # after a tensor can bind that tensor's shape and must stay the same
        # definition after serialization and the native lowering passes.
        if not tvm_ffi.structural_equal(
            self._signature_graph(self.public_abi),
            self._signature_graph(self.abi),
            map_free_vars=True,
        ):
            raise ValueError("Lowered source host changed the public shape/stride parameter graph")

    def raw_module(self, rewrite_call: Callable[[ir.Call, ir.Var], ir.Expr] | None = None):
        parameters, declarations = [], []
        for index, parameter in enumerate(self.entry.params):
            if isinstance(parameter.ty, tirx.TensorType):
                pointer = ir.Var(f"tl_arg_{index}", ir.PtrType(parameter.ty.dtype))
                parameters.append(pointer)
                declarations.append(
                    ir.Bind(
                        parameter,
                        ir.Call("tirx.decl_tensor", [pointer], ty_args=[parameter.ty]),
                        loc=parameter.loc,
                    )
                )
            else:
                parameters.append(parameter)
        parameters.extend(self.abi.implicit_symbols)
        stream = ir.Var("stream", ir.PtrType("void"))
        parameters.append(stream)

        # The retained table is an internal host call graph. Pass the caller's
        # stream through its local helpers, including helpers which only call
        # another helper. Each function owns a distinct parameter identity;
        # helper launches must never fall back to the default device stream.
        streams = {key: ir.Var("stream", ir.PtrType("void")) for key in self.functions if not key.same_as(self.entry_global)}
        streams[self.entry_global] = stream

        def rewrite_body(body, current_stream):
            if body is None:
                return None

            def rewrite_host_call(call):
                result = rewrite_call(call, current_stream) if rewrite_call is not None else call
                if (
                    isinstance(result, ir.Call)
                    and isinstance(result.op, ir.GlobalVar)
                    and result.op in streams
                    and not result.op.same_as(self.entry_global)
                ):
                    return ir.Call(
                        result.op,
                        [*result.args, current_stream],
                        attrs=result.attrs,
                        ty_args=result.ty_args,
                        loc=result.loc,
                        ty=result.ty,
                    )
                return result

            return tvm_ffi.structural_map(body, (ir.Call, rewrite_host_call))

        body = rewrite_body(self.entry.body, stream)

        def rewrite_return(statement):
            value = statement.value
            if (isinstance(value.ty, ir.PrimType) and value.ty == ir.PrimType("void")) or (
                isinstance(value.ty, ir.TupleType) and not value.ty.fields
            ):
                result = ir.Return(ir.const(0, "int32"), loc=statement.loc)
                if isinstance(value, ir.Tuple) and not value.fields:
                    return result
                return ir.SeqStmt([ir.Evaluate(value), result], loc=statement.loc)
            return statement

        body = tvm_ffi.structural_map(body, (ir.Return, rewrite_return))
        raw = tirx.Function(
            parameters,
            [
                ir.Evaluate(tirx.call_extern("__tl_source_begin_call", ty="int32")),
                *declarations,
                *body.seq,
                ir.Return(ir.const(0, "int32")),
            ],
            ir.PrimType("int32"),
            ir.DictAttrs(
                {
                    "global_symbol": "call",
                    "target": Target("c"),
                    "tl.source_entry": True,
                    "tl.source_stream_param_index": len(parameters) - 1,
                }
            ),
            self.entry.loc,
        )
        functions = {}
        for key, function in self.functions.items():
            if key.same_as(self.entry_global):
                continue
            attrs = dict(function.attrs.items()) if function.attrs else {}
            attrs["target"] = Target("c")
            attrs["tl.source_stream_param_index"] = len(function.params)
            functions[key] = tirx.Function(
                [*function.params, streams[key]],
                rewrite_body(function.body, streams[key]),
                function.ret_type,
                ir.DictAttrs(attrs),
                function.loc,
            )
        functions[self.entry_global] = raw
        result = ir.IRModule(functions)
        # Helpers keep their native GlobalVar references. A missing capture is
        # an error, rather than falling back to a name parsed from source text.
        known = set(functions)
        free_globals = set()
        for function in functions.values():
            tvm_ffi.structural_walk(function.body, (ir.GlobalVar, free_globals.add))
        if not free_globals <= known:
            raise ValueError("The native source host table is missing a called helper")
        if not tirx.analysis.check_well_formed(result):
            raise ValueError("Raw source host conversion produced invalid native IR")
        result = tirx.transform.FP8StorageLegalize()(result)
        result = tirx.transform.BF16StorageLegalize()(result)
        # This graph was retained before host finalization. In particular,
        # dynamic launch grids still contain FloorDiv/FloorMod; use the same
        # native intrinsic lowering as the C backend before emitting them.
        result = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(result)
        return result

    def emit(self, *, include_preamble: bool, rewrite_call=None) -> str:
        module = self.raw_module(rewrite_call)
        return str(tvm_ffi.get_global_func("tl.codegen.GenerateSourceHost")(module, include_preamble))

    def emit_device(self, device_module: ir.IRModule, target: Target) -> str:
        """Emit every device launch from the host's native call operands."""
        return str(
            tvm_ffi.get_global_func("tl.codegen.GenerateDeviceSourceHost")(
                self.raw_module(),
                device_module,
                target,
            )
        )


class DeviceSourceWrapper:
    """Native host control flow and the existing device translation unit."""

    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        from tilelang.jit.adapter.wrapper import _require_lowered_modules
        from tilelang.utils.language import retrieve_func_from_module

        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod, self.host_mod = _require_lowered_modules(device_mod, host_mod)
        self.prim_func = retrieve_func_from_module(scheduled_ir_module)
        self.program = SourceHostProgram(self.prim_func, self.host_mod)
        self.srcpath = self.libpath = None
        self.lib_code = source + NATIVE_SOURCE_INIT + self.program.emit_device(self.device_mod, target)
