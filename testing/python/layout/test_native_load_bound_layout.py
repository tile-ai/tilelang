"""Layout bound queries use their own domains when extents read memory."""

import pytest

import tilelang
import tvm_ffi
from tvm_tirx import ir, tirx


def _value(expr, memory):
    if isinstance(expr, ir.IntImm):
        return int(expr.value)
    if isinstance(expr, ir.TensorLoad):
        assert all(_value(index, memory) == 0 for index in expr.indices)
        return memory[expr.source]
    if isinstance(expr, ir.prim.Cast):
        return _value(expr.value, memory)
    operations = {
        ir.prim.Add: lambda a, b: a + b,
        ir.prim.Sub: lambda a, b: a - b,
        ir.prim.Mul: lambda a, b: a * b,
        ir.prim.FloorDiv: lambda a, b: a // b,
        ir.prim.FloorMod: lambda a, b: a % b,
        ir.prim.Min: min,
        ir.prim.Max: max,
    }
    # A leaked layout placeholder has no value in this independent oracle.
    return operations[type(expr)](_value(expr.a, memory), _value(expr.b, memory))


def _bounds():
    return [ir.Var(name, tirx.TensorType((1,), "int32", scope="local.var")) for name in ("a", "b")]


def _assert_no_placeholders(expressions, parameters):
    variables = []
    tvm_ffi.structural_walk(expressions, (ir.Var, variables.append))
    assert all(any(var.same_as(parameter) for parameter in parameters) for var in variables)


@pytest.mark.parametrize("replicate", [1, 2])
def test_load_bound_fragment_capacity_and_thread_extent(replicate):
    a, b = _bounds()
    lanes = 128 // replicate
    forward = (lambda i: (i % lanes, i // lanes)) if replicate == 1 else (lambda i, r: (i % lanes + r * lanes, i // lanes))
    fragment = tilelang.Fragment([b[0] - a[0]], replicate=replicate, forward_fn=forward)
    (capacity,) = fragment.get_output_shape()
    thread_extent = fragment.get_thread_size()
    _assert_no_placeholders([capacity, thread_extent], [a, b])
    for begin, end in ((0, 0), (2, 3), (1, 128), (13, 300), (0, 512)):
        memory = {a: begin, b: end}
        assert _value(capacity, memory) == (end - begin + lanes - 1) // lanes
        assert _value(thread_extent, memory) == 128


def test_load_bound_region_contains_every_touched_coordinate():
    a, b = _bounds()
    layout = tilelang.Layout([512], lambda i: i // 128)
    (region,) = layout.map_region_bounds([ir.Range(a[0], b[0])])
    _assert_no_placeholders([region.begin, region.extent], [a, b])
    for begin, end in ((0, 1), (5, 130), (128, 256), (255, 512)):
        memory = {a: begin, b: end}
        low = _value(region.begin, memory)
        high = low + _value(region.extent, memory)
        assert low == begin // 128
        assert high == (end - 1) // 128 + 1
        assert all(low <= index // 128 < high for index in range(begin, end))
