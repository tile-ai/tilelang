"""Small native workload builder for Carver's built-in templates.

This is a staging graph over real TIRx expressions. It deliberately has no TE
producer objects: inputs, loads, scalar expressions and the emitted Function all
belong to the one native compiler. User-supplied Functions bypass this builder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from tilelang.language._native import encode_annotations
from tilelang.language.dtypes import get_tvm_dtype


def as_expr(value):
    if isinstance(value, ir.Expr):
        return value
    convert = getattr(value, "__tvm_ffi_object__", None)
    if convert is not None:
        return convert()
    return ir.const(value)


def substitute(value, replacements):
    """Substitute expression uses in a value with no local definitions."""
    return tvm_ffi.structural_map(value, (ir.Var, lambda var: replacements.get(var, var)))


def variables(value):
    result = set()
    tvm_ffi.structural_walk(value, (ir.Var, result.add), order="pre")
    return result


def annotation(values):
    schema, payload = encode_annotations(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def attribute(key, value, body):
    schema, payload = encode_annotations({"node": None, "key": key, "value": value})
    return ir.RegionStmt(
        "tl.attr", [payload], [], attrs={"schema_version": 1, "annotation_schema": schema}, body=ir.SeqStmt(body), result_vars=[]
    )


@dataclass(frozen=True)
class IterAxis:
    var: ir.Var
    domain: ir.Range
    kind: str
    step: ir.Expr | None = None


@dataclass(frozen=True)
class Reduction:
    value: ir.Expr
    axes: tuple[IterAxis, ...]
    identity: ir.Expr


@dataclass(frozen=True)
class Stage:
    """An executable stage, retaining native bodies, definitions and guards."""

    uid: int
    name: str
    axes: tuple[IterAxis, ...]
    body: ir.Stmt
    init: ir.Stmt | None = None
    predicate: ir.Expr | None = None
    init_guarded: bool = False

    def substitute(self, replacements):
        return replace(
            self,
            body=substitute(self.body, replacements),
            init=substitute(self.init, replacements) if self.init is not None else None,
            predicate=substitute(self.predicate, replacements) if self.predicate is not None else None,
        )


def emit_stage(stage: Stage):
    statements = [
        annotation({"tl.axis": {"kind": axis.kind, "var": axis.var, "begin": axis.domain.begin, "extent": axis.domain.extent}})
        for axis in stage.axes
    ]
    if stage.init is not None:
        if stage.init_guarded:
            # A parsed native stage already carries the exact If and lexical
            # scope. Re-emission must not accumulate extra guards/wrappers.
            statements.append(stage.init)
        else:
            first = ir.const(True)
            for axis in stage.axes:
                if axis.kind == "R":
                    first = prim.And(first, axis.var == axis.domain.begin)
            statements.append(ir.If(first, attribute("tl.sblock.init", {}, [stage.init]), None))
    statements.append(stage.body)
    body = attribute("tl.sblock", {"name": stage.name, "no_realize": False}, statements)
    if stage.predicate is not None:
        body = ir.If(stage.predicate, body, None)
    for axis in reversed(stage.axes):
        step = axis.step if axis.step is not None else ir.const(1, axis.var.ty.dtype)
        body = ir.For(axis.var, axis.domain.begin, axis.domain.begin + axis.domain.extent, step, ir.ForKind.DEFAULT, ir.SeqStmt([body]))
    return body


def emit_function(parameters, stages, *, attrs=None, prefix=(), loc=None):
    """Emit a stable stage order and allocate only non-parameter temporaries."""
    parameters = list(parameters)
    outputs = []
    for stage in stages:
        tvm_ffi.structural_walk(stage.body, (ir.TensorStore, lambda store: outputs.append(store.dest)), order="pre")
    seen = set(parameters)
    body = list(prefix)
    for stmt in prefix:
        if isinstance(stmt, ir.Bind):
            seen.add(stmt.var)
    for tensor in outputs:
        if tensor not in seen:
            seen.add(tensor)
            body.append(ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty])))
    body.extend(emit_stage(stage) for stage in stages)
    location = {} if loc is None else {"loc": loc}
    function = tirx.Function(parameters, ir.SeqStmt(body), ir.NoneType(), **location)
    for key, value in (attrs or {}).items():
        function = function.with_attr(key, value)
    tirx.analysis.check_well_formed(function)
    return function


class WorkloadBuilder:
    """Construct the six private templates without retaining the old TE stack."""

    def __init__(self, index_dtype="int64"):
        self.index_type = ir.PrimType(index_dtype)
        self.stages: list[Stage] = []
        self._reductions: dict[ir.Var, IterAxis] = {}
        self._inputs = []
        self._next_axis = 0

    def input(self, shape, name="placeholder", dtype="float32"):
        shape = [ir.const(extent, self.index_type.dtype) if isinstance(extent, int) else as_expr(extent) for extent in shape]
        tensor = ir.Var(name, tirx.TensorType(shape, ir.PrimType(get_tvm_dtype(dtype)), "global"))
        self._inputs.append(tensor)
        return tensor

    def reduce_axis(self, domain, name=None):
        if name is None:
            name = f"rv{self._next_axis}"
            self._next_axis += 1
        var = ir.Var(name, self.index_type)
        begin, end = (ir.const(value, self.index_type.dtype) if isinstance(value, int) else as_expr(value) for value in domain)
        self._reductions[var] = IterAxis(var, ir.Range(begin, end), "R")
        return var

    def sum(self, value, axis):
        value = as_expr(value)
        if isinstance(axis, ir.Var):
            axis = [axis]
        axes = tuple(self._reductions[var] for var in axis)
        if not isinstance(value.ty, ir.PrimType):
            raise TypeError("A Carver reduction must have a primitive value")
        return Reduction(value, axes, ir.const(0, value.ty.dtype))

    @staticmethod
    def all(*conditions):
        value = ir.const(True)
        for condition in conditions:
            value = prim.And(value, as_expr(condition))
        return value

    @staticmethod
    def if_then_else(condition, true_value, false_value):
        # if_then_else is lazy. Select may speculatively evaluate an out-of-range
        # convolution load and is therefore a different contract.
        return prim.if_then_else(as_expr(condition), as_expr(true_value), as_expr(false_value))

    def compute(self, shape, fcompute: Callable, name="compute"):
        shape = [ir.const(extent, self.index_type.dtype) if isinstance(extent, int) else as_expr(extent) for extent in shape]
        axes = tuple(
            IterAxis(ir.Var(f"i{index}", self.index_type), ir.Range(ir.const(0, self.index_type.dtype), extent), "S")
            for index, extent in enumerate(shape)
        )
        value = fcompute(*(axis.var for axis in axes))
        reduction = value if isinstance(value, Reduction) else None
        value = reduction.value if reduction is not None else as_expr(value)
        if not isinstance(value.ty, ir.PrimType):
            raise TypeError("A Carver compute must produce a primitive tensor element")
        output = ir.Var(name, tirx.TensorType(shape, value.ty, "global"))
        indices = [axis.var for axis in axes]
        init = None
        if reduction is not None:
            axes = (*axes, *reduction.axes)
            init = ir.TensorStore(output, indices, reduction.identity)
            value = prim.Add(ir.TensorLoad(output, indices), value)
        self.stages.append(Stage(len(self.stages), name, tuple(axes), ir.TensorStore(output, indices, value), init))
        return output

    def to_function(self, parameters: Sequence[ir.Var]):
        # Parameters are an explicit caller contract; unused placeholders such
        # as the optional bias do not leak into it.
        producer = {}
        for stage in self.stages:
            stores = []
            tvm_ffi.structural_walk(stage.body, (ir.TensorStore, stores.append), order="pre")
            for store in stores:
                producer[store.dest] = stage
        selected, visiting = set(), set()

        def include(tensor):
            stage = producer.get(tensor)
            if stage is None or stage.uid in selected:
                return
            if stage.uid in visiting:
                raise ValueError("Cyclic Carver workload graph")
            visiting.add(stage.uid)
            reads = []
            tvm_ffi.structural_walk(stage.body, (ir.TensorLoad, reads.append), order="pre")
            for read in reads:
                if read.source != tensor:
                    include(read.source)
            visiting.remove(stage.uid)
            selected.add(stage.uid)

        for parameter in parameters:
            include(parameter)
        stages = [stage for stage in self.stages if stage.uid in selected]
        return emit_function(parameters, stages, attrs={"global_symbol": "main", "tirx.noalias": True})
