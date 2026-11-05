"""Bounded memory semantics for the original static safe-access fixtures.

This interpreter does not run a compiler pass or an Analyzer. It executes the
declared small integer domains, traps every out-of-bounds dereference, compares
all tensor bytes, and records externally observable atomic/extern operations.
It deliberately rejects unsupported IR; it is not a general TIRx interpreter
or an oracle for parallel schedules, floating-point reductions, or overflow.
"""

from dataclasses import dataclass
import operator

import numpy as np
from tvm_tirx import ir, tirx


@dataclass
class _Pointer:
    memory: np.ndarray
    name: str
    offset: int
    extent: int
    access_mask: int


class _Machine:
    def __init__(self, function, *, scalars=None):
        self.values = {}
        self.memories = []
        self.events = []
        self.steps = 0
        for param in function.params:
            if isinstance(param.ty, tirx.TensorType):
                self.allocate(param)
            else:
                assert isinstance(param.ty, ir.PrimType) and scalars is not None and param in scalars
                self.values[param] = np.asarray(scalars[param], dtype=str(param.ty.dtype))[()]

    def allocate(self, tensor):
        shape = tuple(int(self.expr(dim)) for dim in tensor.ty.shape)
        dtype = np.dtype(str(tensor.ty.dtype))
        # Every address in this bounded fixture must have a distinct, nonzero
        # value in its actual dtype. A short repeating pattern hides bad reads.
        size = int(np.prod(shape))
        data = np.arange(1, size + 1, dtype="int64").astype(dtype).reshape(shape)
        assert np.all(np.isfinite(data)) and np.all(data != 0) and np.unique(data).size == size, (
            "oracle initializer cannot distinguish this dtype/domain",
            dtype,
            shape,
        )
        self.values[tensor] = data
        self.memories.append((tensor.name, data))
        return data

    def indices(self, tensor, expressions):
        data = self.values[tensor]
        indices = tuple(np.asarray(self.expr(index), dtype="int64") for index in expressions)
        assert len(indices) == data.ndim
        for index, extent in zip(indices, data.shape):
            assert np.all((0 <= index) & (index < extent)), (tensor.name, indices, data.shape)  # noqa: SIM300 -- Interval oracle notation.
        return indices

    def pointer(self, load, extent, access_mask):
        assert isinstance(load, ir.TensorLoad)
        data = self.values[load.source]
        indices = [int(self.expr(index)) for index in load.indices]
        # Address formation may be out of bounds; only dereferencing traps.
        offset = 0
        for index, dimension_extent in zip(indices, data.shape):
            offset = offset * dimension_extent + index
        return _Pointer(data.reshape(-1), load.source.name, offset, extent, access_mask)

    @staticmethod
    def check_pointer(pointer, count):
        # The operation determines its actual span. In particular, the public
        # vector-atomic facade starts from an element pointer (extent=1), then
        # the atomic opcode supplies its lane count. Keep that metadata visible
        # without treating it as a separate allocation bound.
        assert count >= 0 and pointer.extent >= 0, pointer
        assert 0 <= pointer.offset and pointer.offset + count <= pointer.memory.size, pointer  # noqa: SIM300 -- Interval oracle notation.

    @staticmethod
    def dereference(pointer, count=1):
        _Machine.check_pointer(pointer, count)
        assert pointer.access_mask & 1, pointer
        result = pointer.memory[pointer.offset : pointer.offset + count]
        return result.copy() if count != 1 else result[0].copy()

    @staticmethod
    def write(pointer, value, count=1):
        _Machine.check_pointer(pointer, count)
        assert pointer.access_mask & 2, pointer
        pointer.memory[pointer.offset : pointer.offset + count] = value

    @staticmethod
    def event_value(value):
        if isinstance(value, _Pointer):
            return (value.name, value.offset, value.extent, value.access_mask)
        array = np.asarray(value)
        return (str(array.dtype), array.shape, array.tobytes())

    def expr(self, node):
        if isinstance(node, (int, float, bool)):
            return node
        if isinstance(node, ir.Var):
            return self.values[node]
        if isinstance(node, (ir.IntImm, ir.FloatImm)):
            return np.asarray(node.value, dtype=str(node.ty.dtype))[()]
        if isinstance(node, ir.StringImm):
            return node.value
        if isinstance(node, ir.TensorLoad):
            return self.values[node.source][self.indices(node.source, node.indices)].copy()
        if isinstance(node, ir.Tuple):
            return tuple(self.expr(item) for item in node.fields)
        if isinstance(node, ir.prim.Ramp):
            return self.expr(node.base) + self.expr(node.stride) * np.arange(int(node.lanes))
        if isinstance(node, ir.prim.Broadcast):
            return np.repeat(self.expr(node.value), int(node.lanes))
        if isinstance(node, ir.prim.Shuffle):
            values = np.concatenate([np.atleast_1d(self.expr(item)) for item in node.vectors])
            return values[[int(self.expr(item)) for item in node.indices]]
        if isinstance(node, ir.prim.Cast):
            return np.asarray(self.expr(node.value)).astype(str(node.ty.dtype).split("x")[0])
        if isinstance(node, ir.prim.Select):
            return np.where(self.expr(node.condition), self.expr(node.true_value), self.expr(node.false_value))
        if isinstance(node, ir.prim.Not):
            return np.logical_not(self.expr(node.a))
        binary = {
            ir.prim.Add: operator.add,
            ir.prim.Sub: operator.sub,
            ir.prim.Mul: operator.mul,
            ir.prim.FloorDiv: operator.floordiv,
            ir.prim.FloorMod: operator.mod,
            ir.prim.LT: operator.lt,
            ir.prim.LE: operator.le,
            ir.prim.GT: operator.gt,
            ir.prim.GE: operator.ge,
            ir.prim.EQ: operator.eq,
            ir.prim.NE: operator.ne,
            ir.prim.And: np.logical_and,
            ir.prim.Or: np.logical_or,
            ir.prim.Min: np.minimum,
            ir.prim.Max: np.maximum,
        }
        if type(node) in binary:
            return binary[type(node)](self.expr(node.a), self.expr(node.b))
        assert isinstance(node, ir.Call) and isinstance(node.op, ir.Op), type(node)
        name = node.op.name
        count = int(node.attrs.get("operand_count", len(node.args))) if node.attrs else len(node.args)
        args = node.args[:count]
        if name == "prim.if_then_else":
            return self.expr(args[1] if bool(self.expr(args[0])) else args[2])
        if name == "tl.access_ptr":
            assert len(args) == 3
            return self.pointer(args[0], int(self.expr(args[1])), int(self.expr(args[2])))
        if name == "tl.annotation":
            return None
        if name in ("tl.ptx_commit_group", "tl.ptx_wait_group", "cuda.ptx_commit_group", "cuda.ptx_wait_group"):
            self.events.append((name, tuple(int(self.expr(arg)) for arg in args)))
            return None
        if name == "tl.ptx_cp_async":
            destination, source = self.expr(args[0]), self.expr(args[1])
            amount = int(self.expr(args[2]))
            enabled = len(args) == 3 or bool(self.expr(args[3]))
            value = self.dereference(source, amount) if enabled else np.zeros(amount, dtype=destination.memory.dtype)
            self.write(destination, value, amount)
            return None
        if name == "tirx.call_extern":
            values = [self.expr(arg) for arg in args]
            self.events.append((values[0], tuple(self.event_value(value) for value in values[1:])))
            # The fixtures discard the handle result; pointer access is their
            # contract. Trap if the guard would expose an invalid pointer.
            for value in values[1:]:
                if isinstance(value, _Pointer):
                    assert value.access_mask in (1, 2, 3), value
                    self.check_pointer(value, value.extent)
            return 0
        if name.startswith("tl.atomic_"):
            values = [self.expr(arg) for arg in args]
            pointer = values[0]
            lanes = 4 if "addx4" in name else 2 if "addx2" in name else 1
            self.events.append((name, tuple(self.event_value(value) for value in values)))
            if "store" in name:
                self.write(pointer, values[1])
                return None
            old = self.dereference(pointer, lanes)
            if "add" in name:
                operand = self.dereference(values[1], lanes) if isinstance(values[1], _Pointer) else values[1]
                self.write(pointer, old + operand, lanes)
            else:
                assert "load" in name, name
            return old
        raise AssertionError(f"Unsupported oracle operator: {name}")

    def stmt(self, node):
        self.steps += 1
        assert self.steps < 200000, "oracle is limited to the declared small fixtures"
        if isinstance(node, ir.SeqStmt):
            for item in node.seq:
                self.stmt(item)
        elif isinstance(node, ir.Bind):
            if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor":
                self.allocate(node.var)
            else:
                self.values[node.var] = self.expr(node.value)
        elif isinstance(node, ir.Evaluate):
            self.expr(node.value)
        elif isinstance(node, ir.TensorStore):
            self.values[node.dest][self.indices(node.dest, node.indices)] = self.expr(node.value)
        elif isinstance(node, ir.For):
            for value in range(int(self.expr(node.begin)), int(self.expr(node.end)), int(self.expr(node.step))):
                self.values[node.var] = value
                self.stmt(node.body)
        elif isinstance(node, ir.If):
            body = node.then_case if bool(self.expr(node.condition)) else node.else_case
            if body is not None:
                self.stmt(body)
        elif isinstance(node, ir.RegionStmt):
            if node.op.name == "tirx.launch_thread":
                assert len(node.body_params) == 1
                for value in range(int(self.expr(node.args[1]))):
                    self.values[node.body_params[0]] = value
                    self.stmt(node.body)
            elif node.op.name == "tl.launch_kernel":
                assert all(int(self.expr(value)) == 0 for value in node.args[0].fields)
                assert all(int(self.expr(value)) == 1 for value in node.args[1].fields)
                assert all(int(self.expr(value)) == 1 for value in node.args[3].fields)
                for variable in node.body_params:
                    self.values[variable] = 0
                self.stmt(node.body)
            else:
                assert node.op.name in ("tl.attr", "tl.lexical_scope"), node.op.name
                self.stmt(node.body)
        else:
            raise AssertionError(f"Unsupported oracle statement: {type(node).__name__}")

    def run(self, function):
        self.stmt(function.body)
        return [(name, str(data.dtype), data.shape, data.tobytes()) for name, data in self.memories], self.events


def assert_same_memory_effects(actual, expected):
    """Compare all bytes and effects after enumerating the fixture's domains."""
    actual_result = _Machine(actual).run(actual)
    expected_result = _Machine(expected).run(expected)
    assert actual_result == expected_result, "Native lowering changed memory bytes or observable effects"


def assert_nonzero_async_padding(actual, original_reference):
    """Check all four elements of each transfer, including its padding tail.

    The original reference program explicitly fills only the first element
    of an invalid four-element transfer. Retain that fixture's valid-copy and
    scalar-padding checks, and add the full transfer oracle required by the
    operation's element-count contract. Do not copy its uninitialized tail.
    """
    machine = _Machine(actual)
    machine.run(actual)
    original = _Machine(original_reference)
    original.run(original_reference)
    assert machine.events == original.events, "Native lowering changed async synchronization effects"
    assert len(machine.memories) == len(original.memories) == 2
    source = machine.memories[0][1]
    output = machine.memories[1][1]
    reference = original.memories[1][1]
    assert source.shape == output.shape == reference.shape == (16,)
    np.testing.assert_array_equal(source, np.arange(1, 17, dtype="float16"))
    np.testing.assert_array_equal(source, original.memories[0][1])
    # The reference already covers the valid copies and the first padding
    # element at each invalid transfer, so these obligations remain explicit.
    np.testing.assert_array_equal(output[:8], reference[:8])
    np.testing.assert_array_equal(output[[8, 12]], reference[[8, 12]])
    np.testing.assert_array_equal(output, np.concatenate([source[8:], np.full(8, 3, dtype="float16")]))


def assert_predicate_table(function, predicate, parameter, cases):
    """Evaluate a scalar guard against independently specified boundary cases."""
    for value, expected in cases:
        machine = _Machine(function, scalars={parameter: value})
        assert bool(machine.expr(predicate)) is expected, (value, expected, predicate)


def assert_nested_lookup(function):
    """Check both lookup levels, including invalid outer and producer indices."""
    a, b, out, index = function.params
    for outer_index in (-1, 0, 1, 2, 3, 4):
        machine = _Machine(function, scalars={index: outer_index})
        machine.values[b][:] = [-1, 0, 7, 8]
        producer = 0 if not 0 <= outer_index < 4 else (-1, 0, 7, 8)[outer_index]
        expected = machine.values[a][producer] if 0 <= producer < 8 else np.float32(0)
        machine.run(function)
        assert machine.values[out][0] == expected, (outer_index, producer, expected)
