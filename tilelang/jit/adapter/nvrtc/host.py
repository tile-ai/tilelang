"""Resolve NVRTC launch operands from host IR, not CUDA parameter names."""

from dataclasses import dataclass

from tilelang import tvm
from tvm import tirx


@dataclass
class HostLaunch:
    name: str
    arguments: list
    bindings: dict
    conditions: tuple


def collect_host_launches(body, parameter_counts):
    """Retain call-site order, lexical bindings, and enclosing conditions.

    Tensor argument extraction remains adapter-owned; this is not an
    interpreter for packed-API validation. Only bindings reachable from
    launch operands are materialized.
    Looped launches are not supported by this wrapper.
    """
    launches = []

    def visit(stmt, bindings, conditions=(), in_loop=False):
        if isinstance(stmt, tirx.SeqStmt):
            local = dict(bindings)
            for child in stmt.seq:
                if isinstance(child, tirx.Bind):
                    local[child.var] = child.value
                else:
                    visit(child, local, conditions, in_loop)
        elif isinstance(stmt, tirx.AttrStmt):
            visit(stmt.body, bindings, conditions, in_loop)
        elif isinstance(stmt, tirx.IfThenElse):
            visit(stmt.then_case, bindings, (*conditions, (stmt.condition, True)), in_loop)
            if stmt.else_case is not None:
                visit(stmt.else_case, bindings, (*conditions, (stmt.condition, False)), in_loop)
        elif isinstance(stmt, (tirx.For, tirx.While)):
            visit(stmt.body, bindings, conditions, True)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if call.op != tvm.ir.Op.get("tirx.tvm_call_packed") or not call.args:
                return
            if not isinstance(call.args[0], tirx.StringImm):
                return
            name = call.args[0].value
            if name not in parameter_counts:
                return
            if in_loop:
                raise ValueError("NVRTC does not support kernel launches inside host loops; use execution_backend='tvm_ffi'.")
            count = parameter_counts[name]
            if len(call.args) < 1 + count:
                raise ValueError(f"NVRTC launch {name} requires {count} arguments, got {len(call.args) - 1}")
            launches.append(HostLaunch(name, list(call.args[1 : 1 + count]), dict(bindings), conditions))

    visit(body, {})
    return launches


class HostScalarEmitter:
    """Emit the supported integer subset, preserving intermediate dtypes."""

    def __init__(self, bindings, inputs, dtype_map, prefix):
        self.bindings = bindings
        self.inputs = inputs
        self.dtype_map = dtype_map
        self.prefix = prefix
        self.values = {}
        self.statements = []
        self.active = set()

    def _cast(self, value, dtype):
        dtype = str(dtype)
        if dtype not in ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
            raise ValueError(f"NVRTC host preparation does not support {dtype}; use execution_backend='tvm_ffi'.")
        return f"{self.dtype_map[dtype]}({value}).value"

    def emit(self, expr):
        if isinstance(expr, int):
            return repr(expr)
        if isinstance(expr, tirx.Var):
            if expr in self.inputs:
                return self.inputs[expr]
            if expr in self.values:
                return self.values[expr]
            if expr not in self.bindings or expr in self.active:
                raise ValueError(f"Cannot resolve NVRTC host argument {expr.name}; use execution_backend='tvm_ffi'.")
            self.active.add(expr)
            rhs = self.emit(self.bindings[expr])
            self.active.remove(expr)
            if str(expr.dtype) == "handle":
                self.values[expr] = rhs
                return rhs
            name = f"{self.prefix}{len(self.values)}"
            self.values[expr] = name
            self.statements.append(f"{name} = {self._cast(rhs, expr.dtype)}")
            return name
        if isinstance(expr, tirx.IntImm):
            return repr(expr.value)
        if isinstance(expr, tirx.Call) and expr.op == tvm.ir.Op.get("tirx.if_then_else"):
            condition = tvm.arith.Analyzer().simplify(expr.args[0])
            if isinstance(condition, tirx.IntImm):
                return self.emit(expr.args[1] if int(condition) else expr.args[2])
            return f"({self.emit(expr.args[1])} if {self.emit(condition)} else {self.emit(expr.args[2])})"
        if isinstance(expr, tirx.Cast):
            return self._cast(self.emit(expr.value), expr.dtype)
        if (
            isinstance(expr, tirx.Call)
            and expr.op == tvm.ir.Op.get("tirx.tvm_struct_get")
            and len(expr.args) == 3
            and isinstance(expr.args[1], tirx.IntImm)
            and int(expr.args[1]) == 0
            and isinstance(expr.args[2], tirx.IntImm)
            and int(expr.args[2]) == 1
            and expr.args[0] in self.inputs
        ):
            # MakePackedAPI extracts DLTensor.data into a fresh host Var.
            return self.inputs[expr.args[0]]
        binary = {
            tirx.Add: "+",
            tirx.Sub: "-",
            tirx.Mul: "*",
            tirx.FloorDiv: "//",
            tirx.FloorMod: "%",
            tirx.LT: "<",
            tirx.LE: "<=",
            tirx.GT: ">",
            tirx.GE: ">=",
            tirx.EQ: "==",
            tirx.NE: "!=",
            tirx.And: "and",
            tirx.Or: "or",
        }
        if type(expr) in binary:
            value = f"({self.emit(expr.a)} {binary[type(expr)]} {self.emit(expr.b)})"
            return self._cast(value, expr.dtype)
        if isinstance(expr, (tirx.Min, tirx.Max)):
            op = "min" if isinstance(expr, tirx.Min) else "max"
            return self._cast(f"{op}({self.emit(expr.a)}, {self.emit(expr.b)})", expr.dtype)
        if isinstance(expr, tirx.Not):
            return f"(not {self.emit(expr.a)})"
        # Buffer loads and calls must not be copied to Python or silently
        # replaced by a same-named argument. Their host semantics need a
        # dedicated lowering (as for TMA descriptors).
        raise ValueError(f"Unsupported NVRTC host expression {type(expr).__name__}; use execution_backend='tvm_ffi'.")

    def take_statements(self):
        statements, self.statements = self.statements, []
        return statements
