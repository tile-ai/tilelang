"""Python hosts over native TIRx graphs and the provider's CUDA launch ABI.

The emitter reads Function, TensorType and CallFFIKernelAttr objects. Generated
modules contain ordinary Python control flow and serializable kernel metadata;
they do not parse printed IR or recover kernel arguments from device source.
CUDA resource limits, clusters and launch attributes remain owned by the same
native runtime used by packed TileLang calls.
"""

from __future__ import annotations

import ctypes
import json
import math
import struct
import sys
from operator import index as integer_index
from pathlib import Path

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from .source_host import SourceHostProgram


def as_pointer(value):
    """Preserve the logical pointer of Tensor, raw pointer and nullable inputs."""
    if isinstance(value, ctypes.c_void_p):
        return value
    if value is None:
        return ctypes.c_void_p()
    if hasattr(value, "data_ptr"):
        return ctypes.c_void_p(value.data_ptr())
    if hasattr(value, "__tvm_ffi_opaque_ptr__"):
        return ctypes.c_void_p(value.__tvm_ffi_opaque_ptr__())
    return ctypes.c_void_p(int(value))


def pointer_offset(value, offset):
    return ctypes.c_void_p((as_pointer(value).value or 0) + int(offset))


def scalar(value, dtype):
    """Evaluate a host primitive at its declared width, including casts."""
    if dtype == "bool":
        return bool(value)
    if dtype.startswith(("int", "uint")):
        unsigned = dtype.startswith("u")
        bits = int(dtype[4:] if unsigned else dtype[3:])
        value = int(value) & ((1 << bits) - 1)
        return value if unsigned or value < (1 << (bits - 1)) else value - (1 << bits)
    if dtype == "float32":
        return ctypes.c_float(value).value
    if dtype == "float64":
        return float(value)
    if dtype == "float16":
        try:
            return struct.unpack("e", struct.pack("e", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)
    raise TypeError(f"Unsupported scalar type in native Python host: {dtype}")


def trunc_div(left, right):
    """Integer division toward zero, without an imprecise float intermediate."""
    quotient = abs(left) // abs(right)
    return -quotient if (left < 0) != (right < 0) else quotient


def trunc_mod(left, right):
    return left - trunc_div(left, right) * right


def select(condition, true_value, false_value):
    # Function argument evaluation preserves Select's eager operands. The
    # separate if_then_else intrinsic uses Python's conditional expression.
    return true_value if condition else false_value


def host_range(begin, end, step, dtype):
    """For follows native ``value < end`` and fixed-width induction semantics."""
    value = scalar(begin, dtype)
    while value < end:
        yield value
        value = scalar(value + step, dtype)


_CTYPES = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}


def host_load(pointer, index, dtype):
    return ctypes.cast(as_pointer(pointer), ctypes.POINTER(_CTYPES[dtype]))[index]


def host_store(pointer, index, value, dtype):
    ctypes.cast(as_pointer(pointer), ctypes.POINTER(_CTYPES[dtype]))[index] = scalar(value, dtype)


class HostResources:
    """Per-invocation storage and stream state, with no cross-call shape facts."""

    def __init__(self, stream, device_id):
        self.stream = as_pointer(stream)
        self.device_id = int(device_id)
        self.storage = []
        self.policies = []

    def __enter__(self):
        self.stream_context = tvm_ffi.use_raw_stream(tvm_ffi.Device(2, self.device_id), self.stream.value or 0)
        self.stream_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            # A native reset removes its matching policy. This fallback only
            # runs when a host exception bypasses the explicit reset node.
            while self.policies:
                stream = self.policies.pop()
                try:
                    tvm_ffi.get_global_func("__tvm_cuda_stream_reset_access_policy_window")(stream)
                except Exception:
                    if exc_type is None:
                        raise
        finally:
            self.stream_context.__exit__(exc_type, exc, traceback)

    def allocate(self, kind, count):
        units = {"tensormap": (128, 64), "shape": (8, 8), "arg_value": (16, 8), "arg_tcode": (4, 4)}
        if kind not in units:
            raise ValueError(f"Unsupported native host stack allocation: {kind}")
        size, alignment = units[kind]
        return self.allocate_bytes(size * int(count), alignment)

    def allocate_bytes(self, size, alignment=64):
        if size < 0:
            raise ValueError("Native host allocation has negative size")
        storage = ctypes.create_string_buffer(int(size) + int(alignment) - 1)
        self.storage.append(storage)
        address = ctypes.addressof(storage)
        return ctypes.c_void_p((address + alignment - 1) // alignment * alignment)

    def packed(self, symbol, *arguments):
        arguments = list(arguments)
        setting = symbol == "__tvm_cuda_stream_set_access_policy_window"
        resetting = symbol == "__tvm_cuda_stream_reset_access_policy_window"
        if setting and len(arguments) < 4:
            if len(arguments) == 2:
                arguments.append(0.8)
            arguments.append(self.stream)
        elif resetting and not arguments:
            arguments.append(self.stream)
        result = tvm_ffi.get_global_func(symbol)(*arguments)
        if setting:
            self.policies.append(as_pointer(arguments[3]))
        elif resetting:
            stream = as_pointer(arguments[0])
            for index in range(len(self.policies) - 1, -1, -1):
                if self.policies[index].value == stream.value:
                    self.policies.pop(index)
                    break
        return result


def _wrap_cuda_kernel_arguments(kernel, argument_types):
    """Convert runtime storage carriers without changing external CUDA types."""
    indices = tuple(index for index, dtype in enumerate(argument_types) if dtype == "uint64")
    boolean_indices = tuple(index for index, dtype in enumerate(argument_types) if dtype == "bool")
    if not indices and not boolean_indices:
        return kernel

    def call(*arguments):
        if len(arguments) < len(argument_types):
            raise TypeError("Missing CUDA kernel arguments")
        arguments = list(arguments)
        for index in boolean_indices:
            # Only raw CUDA source retains a bool8 device signature. The
            # driver reads one byte from this normalized int32 holder; the
            # byte-order requirement is checked while loading its metadata.
            arguments[index] = int(bool(arguments[index]))
        for index in indices:
            value = integer_index(arguments[index])
            if not 0 <= value < 1 << 64:
                raise ValueError(f"CUDA kernel argument {index} value {value} is outside uint64")
            arguments[index] = value if value < 1 << 63 else value - (1 << 64)
        return kernel(*arguments)

    return call


def load_cuda_kernels(binary_path, specs, *, source=None):
    """Load compiled device code using the provider's one launch decoder.

    A symbol can have different call-site field orders. Separate metadata
    wrappers preserve that ordering; identical configurations share a module.
    Each returned packed function retains its module and native resource state.
    """
    payload = Path(binary_path).read_bytes()
    create = tvm_ffi.get_global_func("ffi.Module.create.cuda")
    load_metadata = tvm_ffi.get_global_func("runtime.LoadMetaDataFromJSON")
    groups, assignments = [], {}
    for key, spec in specs.items():
        symbol = spec["symbol"]
        info = {name: value for name, value in spec.items() if name != "symbol"}
        # The frozen runtime accepts signed int64 packed slots but has no
        # uint64 converter. Only its private launch metadata uses int64:
        # CUDA parameters and persistent host specs keep their real uint64
        # types. The callable below supplies the same 64 bits in that slot.
        if "bool" in info["arg_types"] and sys.byteorder != "little":
            raise ValueError("Raw CUDA bool argument holders require little-endian hosts")
        storage_types = {"uint64": "int64", "bool": "int32"}
        info["arg_types"] = [storage_types.get(dtype, dtype) for dtype in info["arg_types"]]
        group_index = next((index for index, group in enumerate(groups) if symbol not in group or group[symbol] == info), None)
        if group_index is None:
            group_index = len(groups)
            groups.append({})
        groups[group_index][symbol] = info
        assignments[key] = group_index, symbol
    modules = [
        create(payload, "cubin", load_metadata(json.dumps({"func_info": group})), {"cuda": source} if source is not None else {})
        for group in groups
    ]
    kernels = {
        key: _wrap_cuda_kernel_arguments(modules[index][symbol], specs[key]["arg_types"]) for key, (index, symbol) in assignments.items()
    }
    return kernels


class NativeDeviceArguments:
    """Shared native shape binding and Torch stream handling for Python hosts."""

    _source_packed_storage = True

    def _initialize_native_arguments(self):
        from tilelang.jit.abi import FunctionABIPlan
        from .abi import source_abi_plan

        FunctionABIPlan(self.prim_func, self.result_idx)
        plan = source_abi_plan(self)
        self.params = plan.validate_parameter_metadata(self.params)
        self.param_dtypes = [parameter.torch_dtype() if parameter.kind == "tensor" else None for parameter in self.params]
        self.param_shapes = [parameter.shape for parameter in self.params]
        self.dynamic_symbolic_order = list(plan.implicit_symbols)
        self.dynamic_symbolic_map = {variable: sources[0].tuple() for variable, sources in plan.sources.items() if sources}

    def _bind_source_arguments(self, plan, arguments):
        return plan.bind_arguments(arguments, packed_storage=self._source_packed_storage)

    def _prepare_native_arguments(self, inputs, *, device):
        """Resolve input metadata and allocate outputs before the raw launch.

        This host step has no CUDA context or stream side effects. The device
        wrapper invokes it inside the selected stream; CPU contract tests can
        exercise the same allocation and binding path without a kernel launch.
        """
        import torch
        from .abi import source_abi_plan

        if len(inputs) + len(self.result_idx) != len(self.params):
            raise ValueError(f"Expected {len(self.params) - len(self.result_idx)} inputs, got {len(inputs)}")
        outputs = set(self.result_idx)
        arguments, cursor = [None] * len(self.params), iter(inputs)
        for index in range(len(arguments)):
            if index not in outputs:
                arguments[index] = next(cursor)
        plan = source_abi_plan(self)
        for index, value in enumerate(arguments):
            if index in plan.tensors and isinstance(value, torch.Tensor) and value.dtype != self.param_dtypes[index]:
                raise TypeError(f"Tensor parameter {index} expected {self.param_dtypes[index]}, got {value.dtype}")
        bindings = self._bind_source_arguments(plan, arguments)
        for index in self.result_idx:
            shape = plan.output_shape(index, bindings, packed_storage=self._source_packed_storage)
            arguments[index] = torch.empty(shape, dtype=self.param_dtypes[index], device=device)
        bindings = self._bind_source_arguments(plan, arguments)
        implicit = [plan.resolve_implicit_value(variable, bindings, require_live_shape=False) for variable in plan.implicit_symbols]
        return arguments, implicit

    def _wrap_native_forward(self, *inputs, stream=None):
        import torch

        if len(inputs) + len(self.result_idx) != len(self.params):
            raise ValueError(f"Expected {len(self.params) - len(self.result_idx)} inputs, got {len(inputs)}")
        tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
        first_tensor = next(iter(tensors), None)
        if first_tensor is not None:
            if not first_tensor.is_cuda or any(value.device != first_tensor.device for value in tensors):
                raise ValueError("CUDA source kernels require all tensor inputs on one CUDA device")
            device = first_tensor.device
        else:
            device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            stream_handle = (
                int(stream.cuda_stream if hasattr(stream, "cuda_stream") else stream)
                if stream is not None
                else int(torch.cuda.current_stream(device).cuda_stream)
            )
            launch_stream = torch.cuda.ExternalStream(stream_handle, device=device)
            with torch.cuda.stream(launch_stream):
                arguments, implicit = self._prepare_native_arguments(inputs, device=device)
                # Retain borrowed inputs on the stream used by raw pointer
                # launches, including a user-supplied noncurrent stream.
                for value in arguments:
                    if isinstance(value, torch.Tensor):
                        value.record_stream(launch_stream)
                self._forward_from_prebuild_lib(*arguments, *implicit, stream=stream_handle, device_id=device.index or 0)
        values = [arguments[index] for index in self.result_idx]
        return values[0] if len(values) == 1 else values


class NativePythonHost:
    """Emit a native host graph without flattening its statement structure."""

    def __init__(self, public_function, host_module, device_module):
        self.program = SourceHostProgram(public_function, host_module)
        self.raw = self.program.raw_module()
        self.functions = dict(self.raw.functions.items())
        self.names = {}
        self.global_names = {global_var: f"_tl_host_{index}" for index, global_var in enumerate(self.functions)}
        self.kernels = {}
        self.function_names = []
        self.device_functions = {}
        for global_var, function in device_module.functions.items():
            if not isinstance(function, tirx.Function):
                raise TypeError("A Python device host requires native device Functions")
            symbol = str(function.attrs.get("global_symbol", global_var.name_hint))
            if symbol in self.device_functions:
                raise ValueError(f"Duplicate device symbol {symbol!r}")
            self.device_functions[symbol] = function
        self.body = []
        self.indent = 0

    def name(self, variable):
        if variable not in self.names:
            self.names[variable] = f"_tl_v_{len(self.names)}"
        return self.names[variable]

    @staticmethod
    def dtype(ty):
        if not isinstance(ty, ir.PrimType) or ty.dtype.lanes != 1:
            raise TypeError(f"Native Python host requires a scalar primitive, got {ty}")
        dtype = str(ty.dtype)
        if dtype not in _CTYPES and dtype != "float16":
            raise TypeError(f"Unsupported primitive in native Python host: {dtype}")
        return dtype

    def cast(self, expression, ty):
        if isinstance(ty, (ir.PtrType, tirx.TensorType)):
            return f"_tl_py.as_pointer({expression})"
        return f"_tl_py.scalar({expression}, {self.dtype(ty)!r})"

    def line(self, text):
        self.body.append("    " * self.indent + text)

    def _tensor_index(self, tensor, indices):
        ty = tensor.ty
        if not isinstance(ty, tirx.TensorType):
            raise TypeError("Native host tensor indexing requires TensorType")
        from tilelang.jit.abi import _tensor_parameter

        descriptor = _tensor_parameter(-1, tensor)
        if len(indices) != len(descriptor.strides):
            raise ValueError("Native host tensor indexing has inconsistent rank")
        offset = descriptor.element_offset
        for index, stride in zip(indices, descriptor.strides):
            offset = offset + index * stride
        return self.expression(offset)

    def _launch(self, call):
        if not call.args or not isinstance(call.args[0], ir.StringImm):
            raise ValueError("Native kernel launch requires a static device symbol")
        symbol = str(call.args[0].value)
        if symbol not in self.device_functions:
            raise ValueError(f"Host launch refers to missing device kernel {symbol!r}")
        function = self.device_functions[symbol]
        attrs = call.attrs
        if not isinstance(attrs, tirx.CallFFIKernelAttr):
            raise TypeError("Native device launch requires CallFFIKernelAttr")
        fields, tags = list(attrs.launch_fields), list(attrs.launch_params)
        if fields and tags:
            raise ValueError("Cannot mix native launch fields and historical tags")
        if len(set(fields or tags)) != len(fields or tags):
            raise ValueError("Native device launch contains duplicate fields")
        flags = {"tirx.use_programtic_dependent_launch", "tirx.use_cooperative_launch", "tirx.use_required_block_dimension"}
        count = len(fields) if fields else sum(tag not in flags for tag in tags)
        if len(call.args) != 1 + len(function.params) + count:
            raise ValueError("Native host launch arity disagrees with the device Function")
        if int(attrs.num_kernel_args) >= 0 and int(attrs.num_kernel_args) != len(function.params):
            raise ValueError("Native launch metadata disagrees with the device Function")
        types, extra = [], []
        for parameter, argument in zip(function.params, call.args[1:]):
            if isinstance(parameter.ty, ir.PtrType):
                if not isinstance(argument.ty, ir.PtrType):
                    raise TypeError("Native device pointer parameter requires a pointer operand")
                types.append("handle")
                extra.append(int(isinstance(parameter.ty.element_type, tirx.TensorMapType)))
            else:
                if parameter.ty != argument.ty:
                    raise TypeError("Native device scalar argument type differs from its parameter")
                types.append(self.dtype(parameter.ty))
                extra.append(0)
        key = f"launch_{len(self.kernels)}"
        self.kernels[key] = {
            "symbol": symbol,
            "name": symbol,
            "arg_types": types,
            "arg_extra_tags": extra,
            "launch_param_tags": tags,
            "cuda_launch_fields": fields,
            "cuda_kernel_attrs": dict(attrs.kernel_attrs),
        }
        if symbol not in self.function_names:
            self.function_names.append(symbol)
        operands = ", ".join(self.expression(argument) for argument in call.args[1:])
        return f"kernels[{key!r}]({operands})"

    def expression(self, expression):
        if isinstance(expression, ir.Var):
            return self.name(expression)
        if isinstance(expression, ir.IntImm):
            return repr(int(expression.value))
        if isinstance(expression, ir.FloatImm):
            value = float(expression.value)
            if math.isnan(value):
                return "float('nan')"
            if math.isinf(value):
                return "float('-inf')" if value < 0 else "float('inf')"
            return repr(value)
        if isinstance(expression, ir.StringImm):
            return repr(str(expression.value))
        if isinstance(expression, prim.Cast):
            return self.cast(self.expression(expression.value), expression.ty)
        if isinstance(expression, prim.Let):
            return f"(lambda {self.name(expression.var)}: {self.expression(expression.body)})({self.expression(expression.value)})"
        if isinstance(expression, ir.Tuple):
            return "(" + "".join(self.expression(field) + "," for field in expression.fields) + ")"
        if isinstance(expression, ir.TupleGetItem):
            return f"({self.expression(expression.tuple_value)})[{int(expression.index)}]"
        if isinstance(expression, ir.TensorLoad):
            index = self._tensor_index(expression.source, expression.indices)
            return f"_tl_py.host_load({self.expression(expression.source)}, {index}, {self.dtype(expression.ty)!r})"
        if isinstance(expression, prim.Select):
            return f"_tl_py.select({self.expression(expression.condition)}, {self.expression(expression.true_value)}, {self.expression(expression.false_value)})"
        if isinstance(expression, (prim.Min, prim.Max)):
            name = "min" if isinstance(expression, prim.Min) else "max"
            return f"{name}({self.expression(expression.a)}, {self.expression(expression.b)})"
        if isinstance(expression, prim.Not):
            return f"(not {self.expression(expression.a)})"
        if isinstance(expression, prim.BitwiseNot):
            return self.cast(f"(~{self.expression(expression.a)})", expression.ty)
        for node, operator in (
            (prim.Add, "+"),
            (prim.Sub, "-"),
            (prim.Mul, "*"),
            (prim.BitwiseAnd, "&"),
            (prim.BitwiseOr, "|"),
            (prim.BitwiseXor, "^"),
            (prim.LShift, "<<"),
            (prim.RShift, ">>"),
            (prim.FloorDiv, "//"),
            (prim.FloorMod, "%"),
            (prim.Div, "/"),
            (prim.Mod, "%"),
            (prim.LT, "<"),
            (prim.LE, "<="),
            (prim.GT, ">"),
            (prim.GE, ">="),
            (prim.EQ, "=="),
            (prim.NE, "!="),
            (prim.And, "and"),
            (prim.Or, "or"),
        ):
            if isinstance(expression, node):
                left, right = self.expression(expression.a), self.expression(expression.b)
                if node in (prim.Div, prim.Mod) and expression.ty.dtype.is_integer:
                    name = "trunc_div" if node is prim.Div else "trunc_mod"
                    result = f"_tl_py.{name}({left}, {right})"
                else:
                    result = f"({left} {operator} {right})"
                return self.cast(result, expression.ty)
        if not isinstance(expression, ir.Call):
            raise TypeError(f"Unsupported node in native Python host: {type(expression).__name__}")
        if isinstance(expression.op, ir.GlobalVar):
            if expression.op not in self.global_names:
                raise ValueError("Native host graph is missing a called helper")
            return f"{self.global_names[expression.op]}({', '.join(self.expression(arg) for arg in expression.args)})"
        name = str(expression.op.name)
        if name == "tirx.call_ffi_kernel":
            return self._launch(expression)
        if name in ("tirx.decl_tensor", "tirx.tensor_data_ptr", "tirx.view"):
            return self.expression(expression.args[0])
        if name in ("tirx.ptr_byte_offset", "tirx.handle_add_byte_offset"):
            return f"_tl_py.pointer_offset({self.expression(expression.args[0])}, {self.expression(expression.args[1])})"
        if name in ("tirx.reinterpret", "ir.reinterpret"):
            value = self.expression(expression.args[0])
            if isinstance(expression.ty, ir.PtrType):
                return f"_tl_py.as_pointer({value})"
            if isinstance(expression.args[0].ty, ir.PtrType):
                return self.cast(f"(_tl_py.as_pointer({value}).value or 0)", expression.ty)
            raise TypeError("Bit reinterpretation between scalar types requires an explicit host lowering")
        if name == "tirx.stack_alloca":
            return f"_tl_resources.allocate({', '.join(self.expression(arg) for arg in expression.args)})"
        if name == "tirx.alloc_tensor":
            ty = expression.ty
            if not isinstance(ty, tirx.TensorType):
                raise TypeError("Host allocation requires TensorType")
            count = " * ".join(self.expression(dim) for dim in ty.shape) or "1"
            bits = ty.dtype.dtype.bits * ty.dtype.dtype.lanes
            return f"_tl_resources.allocate_bytes((({count}) * {bits} + 7) // 8)"
        if name in ("tirx.call_packed", "tirx.call_extern"):
            symbol = expression.args[0]
            if not isinstance(symbol, ir.StringImm):
                raise ValueError("Python host requires a static packed function symbol")
            if str(symbol.value) == "__tl_source_begin_call":
                return "0"
            return f"_tl_resources.packed({', '.join(self.expression(arg) for arg in expression.args)})"
        if name in ("tirx.if_then_else", "ir.if_then_else"):
            cond, yes, no = (self.expression(arg) for arg in expression.args)
            return f"({yes} if {cond} else {no})"
        for operation, spelling in (
            ("bitwise_and", "&"),
            ("bitwise_or", "|"),
            ("bitwise_xor", "^"),
            ("shift_left", "<<"),
            ("shift_right", ">>"),
        ):
            if name in (f"ir.{operation}", f"tirx.{operation}"):
                left, right = (self.expression(arg) for arg in expression.args)
                return self.cast(f"({left} {spelling} {right})", expression.ty)
        if name in ("ir.bitwise_not", "tirx.bitwise_not"):
            return self.cast(f"(~{self.expression(expression.args[0])})", expression.ty)
        if name in ("ir.isnullptr", "tirx.isnullptr"):
            return f"(_tl_py.as_pointer({self.expression(expression.args[0])}).value is None)"
        if name in ("ir.assume", "tirx.assume", "prim.assume"):
            return "None"
        raise ValueError(f"Unsupported native host operation {name!r}; it needs an explicit Python host lowering")

    def statements(self, sequence):
        if not sequence.seq:
            self.line("pass")
            return
        for statement in sequence.seq:
            if isinstance(statement, ir.SeqStmt):
                self.statements(statement)
            elif isinstance(statement, ir.RegionStmt):
                if str(statement.op.name) != "tl.lexical_scope":
                    raise TypeError(f"Unsupported region in native Python host: {statement.op.name}")
                if statement.args or statement.body_params or statement.result_vars:
                    raise ValueError("A TileLang lexical scope has no operands, body parameters, or results")
                if dict(statement.attrs.items()):
                    raise TypeError("A lexical scope with attributes requires explicit Python host lowering")
                # This region executes its body once. Variable identity gives
                # each nested definition a distinct Python name, including
                # same-spelled locals; return/break/continue keep their order.
                self.statements(statement.body)
            elif isinstance(statement, ir.Bind):
                self.line(f"{self.name(statement.var)} = {self.expression(statement.value)}")
            elif isinstance(statement, ir.Evaluate):
                self.line(self.expression(statement.value))
            elif isinstance(statement, ir.Return):
                self.line(f"return {self.expression(statement.value)}")
            elif isinstance(statement, ir.If):
                self.line(f"if {self.expression(statement.condition)}:")
                self.indent += 1
                self.statements(statement.then_case)
                self.indent -= 1
                if statement.else_case is not None:
                    self.line("else:")
                    self.indent += 1
                    self.statements(statement.else_case)
                    self.indent -= 1
            elif isinstance(statement, ir.For):
                if int(statement.kind) == int(ir.ForKind.PARALLEL):
                    raise ValueError("A parallel host loop requires an explicit parallel runtime lowering")
                begin, end, step = (self.expression(value) for value in (statement.begin, statement.end, statement.step))
                self.line(f"for {self.name(statement.var)} in _tl_py.host_range({begin}, {end}, {step}, {self.dtype(statement.var.ty)!r}):")
                self.indent += 1
                self.statements(statement.body)
                self.indent -= 1
            elif isinstance(statement, ir.While):
                self.line(f"while {self.expression(statement.condition)}:")
                self.indent += 1
                self.statements(statement.body)
                self.indent -= 1
            elif isinstance(statement, ir.Break):
                self.line("break")
            elif isinstance(statement, ir.Continue):
                self.line("continue")
            elif isinstance(statement, ir.AssertStmt):
                error = str(statement.error_kind.value)
                if error not in ("ValueError", "TypeError", "RuntimeError", "IndexError", "AssertionError"):
                    error = "RuntimeError"
                message = (
                    " + ".join(f"str({self.expression(part)})" for part in statement.message_parts) or "'Native host assertion failed'"
                )
                self.line(f"if not ({self.expression(statement.condition)}):")
                self.indent += 1
                self.line(f"raise {error}({message})")
                self.indent -= 1
            elif isinstance(statement, ir.TensorStore):
                index = self._tensor_index(statement.dest, statement.indices)
                value = self.expression(statement.value)
                self.line(f"_tl_py.host_store({self.expression(statement.dest)}, {index}, {value}, {self.dtype(statement.value.ty)!r})")
            else:
                raise TypeError(f"Unsupported statement in native Python host: {type(statement).__name__}")

    def emit(self):
        entry = self.functions[self.program.entry_global]
        # The raw ABI includes stream as its final parameter. Public adapter
        # signatures keep stream keyword-only and pass it to that slot here.
        argument_names = [f"_tl_arg_{index}" for index in range(len(entry.params) - 1)]
        signature = ", ".join(["kernels", *argument_names, "*", "stream=0", "device_id=0"])
        self.line(f"def call({signature}):")
        self.indent = 1
        self.line("with _tl_py.HostResources(stream, device_id) as _tl_resources:")
        self.indent = 2
        for global_var, function in self.functions.items():
            self.line(f"def {self.global_names[global_var]}({', '.join(self.name(param) for param in function.params)}):")
            self.indent += 1
            self.statements(function.body)
            self.indent -= 1
        operands = [self.cast(name, parameter.ty) for name, parameter in zip(argument_names, entry.params)]
        operands.append("_tl_py.as_pointer(stream)")
        self.line(f"return {self.global_names[self.program.entry_global]}({', '.join(operands)})")
        header = (
            "from tilelang.jit.adapter import device_py_host as _tl_py\n\n"
            "_native_host_schema = 1\n"
            f"_function_names = {self.function_names!r}\n"
            f"_kernel_specs = {self.kernels!r}\n\n"
        )
        return header + "\n".join(self.body) + "\n"
