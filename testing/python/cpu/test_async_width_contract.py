"""Every injected transaction is legal before any subsequent vectorization."""

import ctypes
import json
import os

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Iter


@pytest.fixture(scope="module", autouse=True)
def existing_native_operator_bridge():
    if tvm_ffi.get_global_func("tl.testing.InjectPTXAsyncCopy", allow_missing=True) is None:
        path = os.environ.get("TILELANG_TEST_NATIVE_PROBE")
        assert path, "Build testing/cpp/native_operator_probe.cc and set TILELANG_TEST_NATIVE_PROBE"
        bridge = ctypes.CDLL(path)
        assert tvm_ffi.get_global_func("tl.testing.InjectPTXAsyncCopy") is not None
        yield bridge
    else:
        yield


def integer(value):
    return ir.IntImm("int32", value)


def tensor(name, scope, dtype="float16", alignment=16, offset=0):
    layout = tirx.TileLayout.from_iters([Iter(128, 1, "m")], [], {"m": offset})
    return ir.Var(name, tirx.TensorType((128,), dtype, scope=scope, layout=layout, data_alignment=alignment))


def nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="post")
    return found


def named_calls(value):
    return [call for call in nodes(value, ir.Call) if isinstance(call.op, ir.Op)]


def cp_calls(value):
    return [call for call in named_calls(value) if call.op.name == "tl.ptx_cp_async"]


class CopyProgram:
    def __init__(self, dtype="float16", source_alignment=16, shared_alignment=16, source_offset=0, shared_offset=0):
        self.source = tensor("source", "global", dtype, source_alignment, source_offset)
        self.shared = tensor("shared", "shared", dtype, shared_alignment, shared_offset)
        self.output = tensor("output", "global", dtype)
        self.lane = ir.Var("lane", "int32")
        self.predicate = ir.Var("predicate", "bool")
        self.offset = ir.Var("offset", "int32")
        self.dtype = dtype

    def copy(self, source_index=None, shared_index=None, predicate=None, fill=0.0):
        source_index = self.lane if source_index is None else source_index
        shared_index = self.lane if shared_index is None else shared_index
        value = ir.TensorLoad(self.source, [source_index])
        if predicate is not None:
            value = ir.Call("prim.if_then_else", [predicate, value, ir.const(fill, self.dtype)])
        return ir.TensorStore(self.shared, [shared_index], value)

    def loop(self, body=None, lanes=8, begin=0, step=1, annotations=None):
        return ir.For(
            self.lane,
            integer(begin),
            integer(begin + lanes * step),
            integer(step),
            ir.ForKind.VECTORIZED,
            [self.copy()] if body is None else body,
            annotations or {},
        )

    def function(self, body):
        return tirx.Function([self.source, self.shared, self.output, self.predicate, self.offset], body, ir.PrimType("void"))


def check(program, body, tmp_path, *, expected_count, explicit=False):
    original = program.function(body)
    assert tirx.analysis.check_well_formed(original)
    saved = ir.save_json(original)
    (tmp_path / "before.json").write_text(saved)
    result = tvm_ffi.get_global_func("tl.testing.InjectPTXAsyncCopy")(original.body, explicit)
    after = program.function(result["statement"])
    assert tirx.analysis.check_well_formed(after)
    assert ir.save_json(original) == saved
    (tmp_path / "injected.json").write_text(ir.save_json(after))
    (tmp_path / "injected.py").write_text(after.script())
    copies = cp_calls(after)
    bits = program.source.ty.dtype.dtype.bits
    widths = [int(call.args[2]) * bits // 8 for call in copies]
    record = {
        "injected": bool(result["injected"]),
        "counts": [int(call.args[2]) for call in copies],
        "widths_bytes": widths,
        "source_alignment": int(program.source.ty.data_alignment),
        "shared_alignment": int(program.shared.ty.data_alignment),
        "explicit_commit_wait_ownership": explicit,
        "input_unchanged": ir.save_json(original) == saved,
    }
    (tmp_path / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    assert all(int(call.args[2]) * bits % 8 == 0 and width in (4, 8, 16) for call, width in zip(copies, widths)), record
    for call in copies:
        call.validate()
    assert len(copies) == (0 if expected_count is None else 1), record
    if expected_count is None:
        assert not result["injected"]
        assert tvm_ffi.structural_equal(original, after)
    else:
        assert result["injected"] and int(copies[0].args[2]) == expected_count

    # This pass explicitly serializes surviving vector loops. Transaction
    # legality must already hold and may not depend on its vectorizer hooks.
    serialized = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(False)(ir.IRModule({"main": after}))["main"]
    assert tirx.analysis.check_well_formed(serialized)
    assert all(loop.kind != ir.ForKind.VECTORIZED for loop in nodes(serialized, ir.For))
    assert [int(call.args[2]) for call in cp_calls(serialized)] == [int(call.args[2]) for call in copies]
    (tmp_path / "vectorization-disabled.json").write_text(ir.save_json(serialized))
    return after


@pytest.mark.parametrize(
    "dtype,lanes",
    [
        ("uint8", 4),
        ("uint8", 8),
        ("uint8", 16),
        ("float16", 2),
        ("float16", 4),
        ("float16", 8),
        ("bfloat16", 2),
        ("bfloat16", 8),
        ("float32", 4),
    ],
)
def test_complete_aligned_copy_emits_one_independently_legal_transaction(dtype, lanes, tmp_path):
    program = CopyProgram(dtype)
    result = check(program, [program.loop(lanes=lanes)], tmp_path, expected_count=lanes)
    assert not nodes(result, ir.For)
    assert [call.op.name for call in named_calls(result) if call.op.name.startswith("tl.ptx_")] == [
        "tl.ptx_cp_async",
        "tl.ptx_commit_group",
        "tl.ptx_wait_group",
    ]


@pytest.mark.parametrize("source_alignment,shared_alignment", [(1, 16), (16, 1), (2, 16), (8, 16)])
def test_weak_actual_alignment_keeps_narrow_copies_synchronous(source_alignment, shared_alignment, tmp_path):
    program = CopyProgram(source_alignment=source_alignment, shared_alignment=shared_alignment)
    check(program, [program.loop()], tmp_path, expected_count=None)


@pytest.mark.parametrize(
    "boundary", ["opaque_sibling", "binding", "if", "annotations", "begin", "step", "strided_source", "strided_shared", "varying_predicate"]
)
def test_narrow_copy_boundaries_retain_the_original_execution(boundary, tmp_path):
    program = CopyProgram()
    loop = program.loop()
    if boundary == "opaque_sibling":
        loop = program.loop([program.copy(), ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")]))])
    elif boundary == "binding":
        temporary = ir.Var("temporary", "int32")
        loop = program.loop([ir.Bind(temporary, program.lane), program.copy()])
    elif boundary == "if":
        loop = program.loop([ir.If(program.predicate, [program.copy()], None)])
    elif boundary == "annotations":
        loop = program.loop(annotations={"pragma_preserve": 1})
    elif boundary == "begin":
        loop = program.loop(begin=1)
    elif boundary == "step":
        loop = program.loop(step=2)
    elif boundary == "strided_source":
        loop = program.loop([program.copy(source_index=program.lane * 2)])
    elif boundary == "strided_shared":
        loop = program.loop([program.copy(shared_index=program.lane * 2)])
    elif boundary == "varying_predicate":
        loop = program.loop([program.copy(predicate=program.lane < 5)])
    check(program, [loop], tmp_path, expected_count=None)


def test_native_constructor_elides_noop_before_injection(tmp_path):
    program = CopyProgram()
    loop = program.loop([program.copy(), ir.Evaluate(integer(0))])
    assert len(loop.body) == 1 and isinstance(loop.body[0], ir.TensorStore)
    check(program, [loop], tmp_path, expected_count=8)


@pytest.mark.parametrize("source_offset,shared_offset,expected", [(0, 0, 8), (8, 0, 8), (0, 8, 8), (1, 0, None), (0, 1, None)])
def test_full_native_layout_offsets_control_transaction_alignment(source_offset, shared_offset, expected, tmp_path):
    program = CopyProgram(source_offset=source_offset, shared_offset=shared_offset)
    check(program, [program.loop()], tmp_path, expected_count=expected)


@pytest.mark.parametrize("known_multiple", [False, True])
def test_symbolic_base_requires_a_proved_transaction_multiple(known_multiple, tmp_path):
    program = CopyProgram()
    base = program.offset * 8 if known_multiple else program.offset
    loop = program.loop([program.copy(source_index=base + program.lane)])
    check(program, [loop], tmp_path, expected_count=8 if known_multiple else None)


@pytest.mark.parametrize("fill,expected", [(0.0, 8), (-0.0, None), (7.0, None)])
def test_uniform_zero_fill_preserves_all_fill_bits(fill, expected, tmp_path):
    program = CopyProgram()
    result = check(program, [program.loop([program.copy(predicate=program.predicate, fill=fill)])], tmp_path, expected_count=expected)
    if expected is not None:
        assert cp_calls(result)[0].args[3].same_as(program.predicate)


@pytest.mark.parametrize("explicit", [False, True])
def test_complete_copy_preserves_commit_wait_ownership_before_consumer(explicit, tmp_path):
    program = CopyProgram()
    consume = ir.TensorStore(program.output, [integer(0)], ir.TensorLoad(program.shared, [integer(0)]))
    result = check(program, [program.loop(), consume], tmp_path, expected_count=8, explicit=explicit)
    statements = list(result.body)
    assert isinstance(statements[-1], ir.TensorStore)
    events = [statement.value.op.name for statement in statements[:-1]]
    assert events == (["tl.ptx_cp_async"] if explicit else ["tl.ptx_cp_async", "tl.ptx_commit_group", "tl.ptx_wait_group"])


@pytest.mark.parametrize("lanes", [1, 3, 16])
def test_illegal_total_width_keeps_each_half_copy_synchronous(lanes, tmp_path):
    program = CopyProgram()
    check(program, [program.loop(lanes=lanes)], tmp_path, expected_count=None)
