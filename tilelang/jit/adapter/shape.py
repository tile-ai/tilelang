"""Resolve output extents from a PrimFunc's runtime arguments."""

from __future__ import annotations

import operator

import torch
from tvm import arith, tirx
from tvm.error import TVMError


def _stride_scale(buffer):
    bits = buffer.dtype.bits * buffer.dtype.lanes
    return 8 // bits if bits < 8 else 1


class OutputShapeResolver:
    """Evaluate logical output shapes without depending on cached Var identities.

    Shapes and bindings both come from the original PrimFunc. Only input
    tensors and explicit integer scalars can supply values; an unallocated
    output is never used to infer its own shape.
    """

    def __init__(self, func, result_idx, *, shape_scale=None, stride_scale=_stride_scale):
        self.shapes = {}
        self.names = {}
        self.sources = {}
        self.variables = {}
        if not result_idx:
            return

        for i, param in enumerate(func.params):
            if i in result_idx:
                buffer = func.buffer_map[param]
                self.shapes[i] = [int(dim) if isinstance(dim, tirx.IntImm) else dim for dim in buffer.shape]
                self.names[i] = buffer.name
            elif param not in func.buffer_map and param.dtype.type_code in (0, 1):
                self.sources[param] = [(2, i, -1, 1)]

        # Prefer a shape carrier over a stride carrier, retaining alternatives
        # for optional inputs. Scalar parameters take precedence over both.
        for kind in (0, 1):
            for i, param in enumerate(func.params):
                if i in result_idx or param not in func.buffer_map:
                    continue
                buffer = func.buffer_map[param]
                dimensions = buffer.shape if kind == 0 else buffer.strides
                for j, dim in enumerate(dimensions):
                    if isinstance(dim, tirx.Var):
                        scale = (shape_scale(buffer, j) if shape_scale else 1) if kind == 0 else stride_scale(buffer)
                        self.sources.setdefault(dim, []).append((kind, i, j, scale))

        for i, shape in self.shapes.items():
            for j, expr in enumerate(shape):
                if isinstance(expr, int):
                    continue
                variables = []
                tirx.stmt_functor.post_order_visit(
                    expr, lambda node, variables=variables: variables.append(node) if isinstance(node, tirx.Var) else None
                )
                self.variables[i, j] = variables

    def _resolve_var(self, var, values):
        for kind, index, dimension, scale in self.sources.get(var, ()):
            value = values[index]
            if kind == 2:
                if isinstance(value, torch.Tensor):
                    raise TypeError(f"symbol '{var}' requires an integer scalar, not a tensor")
                return operator.index(value)
            if isinstance(value, torch.Tensor):
                return (value.shape[dimension] if kind == 0 else value.stride(dimension)) * scale
        raise ValueError(f"symbol '{var}' has no live input tensor or integer scalar source")

    def __call__(self, index, values):
        shape = []
        resolved = {}
        analyzer = None
        for dimension, expr in enumerate(self.shapes[index]):
            try:
                if isinstance(expr, int):
                    value = expr
                elif isinstance(expr, tirx.Var):
                    value = self._resolve_var(expr, values)
                else:
                    for var in self.variables[index, dimension]:
                        if var not in resolved:
                            resolved[var] = tirx.IntImm(var.dtype, self._resolve_var(var, values))
                    if analyzer is None:
                        analyzer = arith.Analyzer()
                    evaluated = analyzer.simplify(tirx.stmt_functor.substitute(expr, resolved))
                    if not isinstance(evaluated, tirx.IntImm):
                        raise ValueError("expression did not resolve to an integer")
                    value = int(evaluated)
                if value < 0:
                    raise ValueError(f"negative extent {value}")
            except (TypeError, ValueError, OverflowError, TVMError) as error:
                raise ValueError(f"Cannot resolve output {self.names[index]} dimension {dimension} ({expr}): {error}") from error
            shape.append(value)
        return shape
