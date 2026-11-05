"""Cast planning preserves each conversion's source, random bits, and effects.

The finite oracle records the inputs to each conversion. It does not implement
stochastic rounding or use the planner's analyzer to derive expected values.
All arithmetic stays in range, and tensors use one dimensional default layouts.
"""

import json
import operator
from typing import ClassVar

import pytest
import tilelang
import tilelang.language as T
import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

CASES = [
    ("scalar", 0, 4),
    ("scalar", 2, 4),
    ("shared", 0, 4),
    ("group4", 0, 4),
    ("group4", 2, 2),
    ("group4", 3, 1),
    ("group2", 0, 2),
    ("group2", 2, 2),
    ("group2", 3, 1),
    ("element", 0, 1),
    ("rng", 0, 1),
    ("erased_rng", 0, 1),
    ("bound_scalar", 0, 4),
    ("bound_shared", 0, 4),
    ("bound_group4", 0, 4),
    ("bound_group2", 0, 2),
    ("bound_element", 0, 1),
    ("bound_rng", 0, 1),
    ("bound_erased_rng", 0, 1),
    ("chain_group2", 0, 2),
    ("chain_element", 0, 1),
    ("outer_rng", 0, 4),
    ("outer_load", 0, 4),
]


def _objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append))
    return result


def _module(mode, begin):
    def tensor(name, dtype):
        return ir.Var(name, tirx.TensorType([64], dtype, scope="global", layout="default", data_alignment=64))

    source, output, random = tensor("A", "float32"), tensor("B", "float16"), tensor("R", "uint32")
    lane = ir.Var("i", "int32")
    local, outer = [], []
    shape = mode.removeprefix("bound_").removeprefix("chain_")
    bits = ir.const(0x12345678, "uint32")
    if shape in ("shared", "outer_load"):
        bits = random[0]
    elif shape in ("group4", "group2"):
        bits = random[lane // int(shape[-1])]
    elif shape == "element":
        bits = random[lane]
    elif shape in ("rng", "erased_rng", "outer_rng"):
        bits = T.rng_rand()
        if shape == "erased_rng":
            # Keep the evaluated effect in the input, even though the result
            # is algebraically independent of the induction variable.
            bits = prim.Mul(bits, ir.const(0, "uint32"))
    else:
        assert shape == "scalar"
    if mode.startswith(("bound_", "chain_", "outer_")):
        variable = ir.Var("random_bits", bits.ty)
        (outer if mode.startswith("outer_") else local).append(ir.Bind(variable, bits))
        bits = variable
    if mode.startswith("chain_"):
        alias = ir.Var("random_bits_alias", bits.ty)
        local.append(ir.Bind(alias, bits))
        bits = alias
    call = ir.Call(
        "tl.cast",
        [source[lane - begin], bits],
        attrs={"round": "rs", "sat": True},
        ty_args=[ir.PrimType("float16")],
    )
    loop = ir.For(
        lane,
        begin,
        begin + 8,
        1,
        ir.ForKind.VECTORIZED,
        [*local, ir.TensorStore(output, [lane - begin], call)],
    )
    target = Target({"kind": "cuda", "arch": "sm_100a"})
    # The oracle's A/B/R buffers are separate allocations.
    function = tirx.Function([source, output, random], [*outer, loop]).with_attrs({"target": target, "tirx.noalias": True})
    return ir.IRModule({"main": function}), target


def _validate(module):
    function = module["main"]
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    for call in _objects(function, ir.Call):
        call.validate()


class _ConversionInputs:
    _binary: ClassVar[dict] = {
        prim.Add: operator.add,
        prim.Sub: operator.sub,
        prim.Mul: operator.mul,
        prim.FloorDiv: operator.floordiv,
        prim.FloorMod: operator.mod,
    }

    def __init__(self):
        self.variables = {}
        self.memory = {
            "A": [i * 0.25 for i in range(64)],
            "B": [None] * 64,
            "R": [0x10203040 + i for i in range(64)],
        }
        self.rng = []

    @staticmethod
    def _map(operation, *values):
        widths = {len(value) for value in values if isinstance(value, list)}
        if not widths:
            return operation(*values)
        assert len(widths) == 1
        width = widths.pop()
        expanded = [value if isinstance(value, list) else [value] * width for value in values]
        return [operation(*lanes) for lanes in zip(*expanded)]

    def expr(self, value):
        if isinstance(value, (ir.IntImm, ir.FloatImm)):
            return value.value
        if isinstance(value, ir.Var):
            return self.variables[value]
        for kind, operation in self._binary.items():
            if isinstance(value, kind):
                return self._map(operation, self.expr(value.a), self.expr(value.b))
        if isinstance(value, prim.Cast):
            return self.expr(value.value)
        if isinstance(value, prim.Ramp):
            base, stride = self.expr(value.base), self.expr(value.stride)
            return [base + lane * stride for lane in range(value.ty.dtype.lanes)]
        if isinstance(value, prim.Broadcast):
            return [self.expr(value.value)] * value.ty.dtype.lanes
        if isinstance(value, ir.TensorLoad):
            assert len(value.indices) == 1
            return self._map(lambda index: self.memory[value.source.name][index], self.expr(value.indices[0]))
        assert isinstance(value, ir.Call), type(value)
        if value.op.name == "tl.rng_rand":
            sample = 0x34567000 + len(self.rng)
            self.rng.append(sample)
            return sample
        assert value.op.name == "tl.cast", value.op.name
        data, bits = (self.expr(arg) for arg in value.args)
        assert not isinstance(bits, list), "One packed conversion accepts scalar random bits"
        return self._map(lambda item: (item, bits), data)

    def stmt(self, value):
        if isinstance(value, ir.SeqStmt):
            for child in value.seq:
                self.stmt(child)
        elif isinstance(value, ir.For):
            for index in range(self.expr(value.begin), self.expr(value.end), self.expr(value.step)):
                self.variables[value.var] = index
                self.stmt(value.body)
            self.variables.pop(value.var)
        elif isinstance(value, ir.Bind):
            self.variables[value.var] = self.expr(value.value)
        elif isinstance(value, ir.TensorStore):
            data = self.expr(value.value)
            indices = self.expr(value.indices[0])
            if not isinstance(indices, list):
                indices, data = [indices], [data]
            assert len(indices) == len(data)
            for index, item in zip(indices, data):
                assert 0 <= index < 64
                self.memory[value.dest.name][index] = item
        else:
            raise AssertionError(type(value))

    def run(self, module):
        self.stmt(module["main"].body)
        return self.memory["B"], self.rng


def _planned(module, target, expected_width, tmp_path):
    _validate(module)
    before = ir.save_json(module)
    # Pin the supported 128-bit policy so widths describe random-bit sharing,
    # independent of this target's optional 256-bit global memory vectors.
    with target, ir.transform.PassContext(config={"tl.disable_vectorize_256": True}):
        planned = tilelang.transform.LegalizeVectorizedLoop()(module)
    assert ir.save_json(module) == before
    _validate(planned)
    widths = [int(loop.end.value - loop.begin.value) for loop in _objects(planned, ir.For) if loop.kind == ir.ForKind.VECTORIZED]
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "planned.json").write_text(ir.save_json(planned))
    (tmp_path / "planned.txt").write_text(str(planned))
    expected = _ConversionInputs().run(module)
    actual = _ConversionInputs().run(planned)
    (tmp_path / "semantics.json").write_text(json.dumps({"before": expected, "planned": actual, "widths": widths}, indent=2))
    assert actual == expected
    assert widths == ([expected_width] if expected_width > 1 else [])
    return planned


@pytest.mark.parametrize("mode,begin,expected_width", CASES)
def test_cast_plan_preserves_random_bit_scope(mode, begin, expected_width, tmp_path):
    module, target = _module(mode, begin)
    _planned(module, target, expected_width, tmp_path)


@pytest.mark.parametrize(
    "mode,begin,expected_width",
    [
        ("scalar", 2, 4),
        ("shared", 0, 4),
        ("group4", 0, 4),
        ("group2", 2, 2),
        ("element", 0, 1),
        ("rng", 0, 1),
        ("bound_element", 0, 1),
        ("bound_rng", 0, 1),
        ("outer_rng", 0, 4),
    ],
)
def test_cast_emission_preserves_each_conversion_input(mode, begin, expected_width, tmp_path):
    module, target = _module(mode, begin)
    planned = _planned(module, target, expected_width, tmp_path)
    frozen = ir.save_json(planned)
    with target:
        emitted = tilelang.transform.VectorizeLoop()(planned)
    assert ir.save_json(planned) == frozen
    _validate(emitted)
    (tmp_path / "emitted.json").write_text(ir.save_json(emitted))
    (tmp_path / "emitted.txt").write_text(str(emitted))
    expected = _ConversionInputs().run(module)
    actual = _ConversionInputs().run(emitted)
    (tmp_path / "emitted-semantics.json").write_text(json.dumps({"before": expected, "emitted": actual}, indent=2))
    assert actual == expected
    casts = [call for call in _objects(emitted, ir.Call) if call.op.name == "tl.cast"]
    assert len(casts) == 1
    assert casts[0].ty.dtype.lanes == expected_width
    assert casts[0].args[1].ty == ir.PrimType("uint32")
