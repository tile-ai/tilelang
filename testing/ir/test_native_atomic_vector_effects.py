"""Packing atomic pointers must preserve evaluated address effects and values.

The finite-domain oracle below interprets the actual input and transformed IR.
It deliberately does not use the analyzer, pointer resolver, or layout lowering
under test to derive addresses. All tensors in these cases are one dimensional
contiguous allocations, and all arithmetic is in range for its declared type.
"""

import operator

import pytest
import tilelang
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


def _tensor(name, dtype="float32"):
    return ir.Var(
        name,
        tirx.TensorType((64,), dtype, scope="global", layout="default", data_alignment=64),
    )


def _pointer(tensor, index, kind):
    if kind == "address_of":
        return tirx.address_of(tensor[index])
    if kind == "ptr_access":
        return ir.Call("tl.ptr_access", [tensor.data, index, ir.const(1), ir.const(3)])
    if kind == "byte_offset":
        return ir.Call("tirx.ptr_byte_offset", [tensor.data, index * 4], ty=tensor.data.ty)
    if kind == "nested_reinterpret":
        typed = tirx.reinterpret(tensor.data, ty=ir.PtrType(ir.PrimType("float32")))
        shifted = ir.Call("tirx.ptr_byte_offset", [typed, index * 4], ty=typed.ty)
        return ir.Call("tl.ptr_access", [shifted, ir.const(0), ir.const(1), ir.const(3)])
    ptr = ir.Call("tl.access_ptr", [tensor[index], ir.const(1), ir.const(3)])
    if kind == "reinterpret":
        return tirx.reinterpret(ptr, ty=ir.PtrType(ir.PrimType("float32")))
    assert kind == "access"
    return ptr


def _module(kind, pure, *, effect_in_source=False, erased_by_zero_stride=False):
    source = _tensor("A")
    dest = _tensor("B", "int32" if "reinterpret" in kind else "float32")
    lane = ir.Var("lane", "int32")
    base = (tirx.call_pure_extern if pure else tirx.call_extern)("next_base", ty="int32")
    # Explicit nodes retain the evaluated call even in the zero-multiplier
    # counterexample; the oracle checks that it really exists in the input.
    index = prim.Add(prim.Mul(base, ir.const(0 if erased_by_zero_stride else 4)), lane)
    pointer = _pointer(dest, lane if effect_in_source else index, kind)
    contribution = source[index if effect_in_source else lane]
    atomic = ir.Call("tl.atomic_add_elem_op", [pointer, contribution, ir.const(0)])
    loop = ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, ir.Evaluate(atomic))
    target = Target({"kind": "cuda", "arch": "sm_90"})
    # The finite oracle uses independent A/B allocations.
    func = tirx.Function([source, dest], loop).with_attrs({"target": target, "tirx.noalias": True})
    return ir.IRModule({"main": func}), target


def _objects(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append))
    return result


def _validate(module):
    function = module["main"]
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    for call in _objects(function, ir.Call):
        call.validate()


def _transform(module, target, route):
    before = ir.save_json(module)
    _validate(module)
    with target:
        if route == "planned":
            planned = tilelang.transform.LegalizeVectorizedLoop()(module)
            _validate(planned)
            emitted = tilelang.transform.VectorizeLoop()(planned)
        else:
            assert route == "direct"
            planned = None
            emitted = tilelang.transform.VectorizeLoop()(module)
    assert ir.save_json(module) == before
    _validate(emitted)
    return planned, emitted


class _AddressOracle:
    """Execute the small straight-line/loop subset, recording observable events.

    An opaque next_base is either a counter or a constant with an observable
    call. A pure next_base returns 2 and has no observable event. Tensor A[j]
    contains j+1; B starts at zero. Typed pointer wrappers use byte addresses.
    This is a transformation oracle, not a claim of CUDA device execution.
    """

    _binary = {
        prim.Add: operator.add,
        prim.Sub: operator.sub,
        prim.Mul: operator.mul,
        prim.FloorDiv: operator.floordiv,
        prim.FloorMod: operator.mod,
    }

    def __init__(self, behavior):
        self.behavior = behavior
        self.count = 0
        self.variables = {}
        self.events = []
        self.memory = {"A": list(range(1, 65)), "B": [0] * 64}

    def _load_address(self, load):
        assert isinstance(load.source.ty, tirx.TensorType)
        assert len(load.indices) == 1
        # These input allocations have 32-bit elements and default layout.
        return str(load.source.name), self.expr(load.indices[0]) * 4

    def _read(self, address):
        name, offset = address
        assert offset % 4 == 0 and 0 <= offset < 64 * 4
        return self.memory[name][offset // 4]

    def expr(self, value):
        if isinstance(value, ir.IntImm):
            return int(value.value)
        if isinstance(value, ir.Var):
            return self.variables[value]
        for cls, operation in self._binary.items():
            if isinstance(value, cls):
                return operation(self.expr(value.a), self.expr(value.b))
        if isinstance(value, prim.Cast):
            return self.expr(value.value)
        if isinstance(value, ir.TensorLoad):
            return self._read(self._load_address(value))
        assert isinstance(value, ir.Call), type(value)
        name = value.op.name
        if name in ("tirx.call_extern", "tirx.call_pure_extern"):
            assert str(value.args[0].value) == "next_base" and len(value.args) == 1
            if name == "tirx.call_pure_extern":
                return 2
            result = self.count if self.behavior == "counter" else 0
            self.count += 1
            self.events.append(("extern", "next_base", result))
            return result
        if name in ("tl.access_ptr", "tirx.address_of"):
            address = self._load_address(value.args[0])
            if name == "tl.access_ptr":
                self.expr(value.args[1])
                self.expr(value.args[2])
            return address
        if name == "tirx.tensor_data_ptr":
            return str(value.args[0].name), 0
        if name == "tirx.reinterpret":
            return self.expr(value.args[0])
        if name in ("tl.ptr_access", "tirx.ptr_byte_offset"):
            root, offset = self.expr(value.args[0])
            delta = self.expr(value.args[1])
            if name == "tl.ptr_access":
                self.expr(value.args[2])
                self.expr(value.args[3])
                delta *= 4
            return root, offset + delta
        if name in ("tl.atomic_add_elem_op", "tl.atomic_addx4_elem_op"):
            destination = self.expr(value.args[0])
            source = self.expr(value.args[1])
            assert self.expr(value.args[2]) == 0
            lanes = 4 if name == "tl.atomic_addx4_elem_op" else 1
            for lane in range(lanes):
                root, offset = destination[0], destination[1] + lane * 4
                contribution = self._read((source[0], source[1] + lane * 4)) if lanes == 4 else source
                assert offset % 4 == 0 and 0 <= offset < 64 * 4
                self.memory[root][offset // 4] += contribution
                self.events.append(("atomic", root, offset, contribution))
            return None
        raise AssertionError(f"Unexpected operator in finite oracle: {name}")

    def stmt(self, value):
        if isinstance(value, ir.SeqStmt):
            for child in value.seq:
                self.stmt(child)
        elif isinstance(value, ir.For):
            for lane in range(self.expr(value.begin), self.expr(value.end), self.expr(value.step)):
                self.variables[value.var] = lane
                self.stmt(value.body)
            self.variables.pop(value.var)
        elif isinstance(value, ir.Bind):
            self.variables[value.var] = self.expr(value.value)
        elif isinstance(value, ir.Evaluate):
            self.expr(value.value)
        else:
            raise AssertionError(f"Unexpected statement in finite oracle: {type(value)}")

    def run(self, module):
        self.stmt(module["main"].body)
        return self.events, self.memory["B"]


@pytest.mark.parametrize("route", ["planned", "direct"])
@pytest.mark.parametrize("kind", ["access", "address_of", "reinterpret", "ptr_access", "byte_offset", "nested_reinterpret"])
@pytest.mark.parametrize("behavior", ["counter", "constant", "pure"])
def test_atomic_address_effects_keep_their_events_and_addresses(route, kind, behavior):
    original, target = _module(kind, behavior == "pure")
    expected = _AddressOracle(behavior).run(original)
    effects = [event for event in expected[0] if event[0] == "extern"]
    addresses = [event[2] // 4 for event in expected[0] if event[0] == "atomic"]
    assert len(effects) == (0 if behavior == "pure" else 4)
    assert addresses == {"counter": [0, 5, 10, 15], "constant": [0, 1, 2, 3], "pure": [8, 9, 10, 11]}[behavior]
    planned, emitted = _transform(original, target, route)
    assert _AddressOracle(behavior).run(emitted) == expected
    loops = _objects(emitted, ir.For)
    if behavior == "pure":
        assert not loops
        assert any(call.op.name == "tl.atomic_addx4_elem_op" for call in _objects(emitted, ir.Call))
    else:
        assert len(loops) == 1 and loops[0].kind == ir.ForKind.DEFAULT
        if planned is not None:
            assert all(loop.kind != ir.ForKind.VECTORIZED for loop in _objects(planned, ir.For))


@pytest.mark.parametrize("route", ["planned", "direct"])
@pytest.mark.parametrize("behavior", ["counter", "constant", "pure"])
def test_atomic_source_index_keeps_its_effects_and_contributions(route, behavior):
    original, target = _module("access", behavior == "pure", effect_in_source=True)
    expected = _AddressOracle(behavior).run(original)
    contributions = [event[3] for event in expected[0] if event[0] == "atomic"]
    assert contributions == {"counter": [1, 6, 11, 16], "constant": [1, 2, 3, 4], "pure": [9, 10, 11, 12]}[behavior]
    planned, emitted = _transform(original, target, route)
    assert _AddressOracle(behavior).run(emitted) == expected
    if behavior == "pure":
        assert not _objects(emitted, ir.For)
    elif planned is not None:
        assert all(loop.kind != ir.ForKind.VECTORIZED for loop in _objects(planned, ir.For))


@pytest.mark.parametrize("route", ["planned", "direct"])
def test_atomic_effect_is_checked_before_address_algebra_can_discard_it(route):
    original, target = _module("reinterpret", False, erased_by_zero_stride=True)
    expected = _AddressOracle("counter").run(original)
    assert [event[2] for event in expected[0] if event[0] == "extern"] == [0, 1, 2, 3]
    assert [event[2] // 4 for event in expected[0] if event[0] == "atomic"] == [0, 1, 2, 3]
    _, emitted = _transform(original, target, route)
    assert _AddressOracle("counter").run(emitted) == expected
