"""Ragged fragments preserve every logical element's thread and local slot."""

from collections import Counter
from itertools import product
import json
import operator

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.layout import Fragment

from test_native_reducer_contracts import allocate, annotation, annotations, region, tensor, tileop
from test_native_webgpu_ops import _function, _serialize_launches


def target(kind):
    options = {"kind": kind}
    if kind == "cuda":
        options["arch"] = "sm_80"
    return tvm_tirx.target.Target(options)


def _number(expression, bindings, load=None):
    if isinstance(expression, (ir.IntImm, ir.FloatImm)):
        return expression.value
    if isinstance(expression, ir.Var):
        return bindings[expression]
    if isinstance(expression, ir.TensorLoad):
        assert load is not None
        return load(expression.source, tuple(_number(index, bindings, load) for index in expression.indices))
    kind = type(expression).__name__
    if kind == "Cast":
        value = _number(expression.value, bindings, load)
        return float(value) if str(expression.ty.dtype).startswith("float") else int(value)
    if kind == "Select":
        branch = expression.true_value if _number(expression.condition, bindings, load) else expression.false_value
        return _number(branch, bindings, load)
    if kind == "Not":
        return not _number(expression.a, bindings, load)
    binary = {
        "Add": operator.add,
        "Sub": operator.sub,
        "Mul": operator.mul,
        "FloorDiv": operator.floordiv,
        "FloorMod": operator.mod,
        "LT": operator.lt,
        "LE": operator.le,
        "GT": operator.gt,
        "GE": operator.ge,
        "EQ": operator.eq,
        "NE": operator.ne,
        "And": lambda a, b: bool(a and b),
        "Or": lambda a, b: bool(a or b),
        "Min": min,
        "Max": max,
    }
    assert kind in binary, (kind, expression)
    return binary[kind](_number(expression.a, bindings, load), _number(expression.b, bindings, load))


def _enumerate_assignments(function, source, output, thread, shape):
    """Interpret all finite native loops and guards, without any inverse API."""
    memory, bindings = {}, {}
    writes, reads, output_trace = [], [], []
    for ordinal, indices in enumerate(product(*(range(size) for size in shape))):
        memory[(source, None, indices)] = ordinal + 10

    def local(value):
        return value.ty.scope.name.startswith("local")

    def cell(value, indices):
        return value, bindings[thread] if local(value) else None, indices

    def load(value, indices):
        key = cell(value, indices)
        reads.append(key)
        assert key in memory, ("uninitialized or out-of-bounds read", key)
        return memory[key]

    def visit(statement):
        if isinstance(statement, ir.SeqStmt):
            for child in statement.seq:
                visit(child)
        elif isinstance(statement, ir.RegionStmt):
            assert statement.op.name == "tirx.launch_thread"
            assert len(statement.body_params) == 1
            for index in range(_number(statement.args[1], bindings)):
                bindings[statement.body_params[0]] = index
                visit(statement.body)
        elif isinstance(statement, ir.For):
            for index in range(_number(statement.begin, bindings), _number(statement.end, bindings), _number(statement.step, bindings)):
                bindings[statement.var] = index
                visit(statement.body)
        elif isinstance(statement, ir.If):
            if _number(statement.condition, bindings):
                visit(statement.then_case)
            elif statement.else_case is not None:
                visit(statement.else_case)
        elif isinstance(statement, ir.Bind):
            if isinstance(statement.var.ty, tirx.TensorType):
                assert isinstance(statement.value, ir.Call) and statement.value.op.name == "tirx.alloc_tensor"
            else:
                bindings[statement.var] = _number(statement.value, bindings, load)
        elif isinstance(statement, ir.TensorStore):
            indices = tuple(_number(index, bindings) for index in statement.indices)
            assert all(0 <= index < _number(extent, bindings) for index, extent in zip(indices, statement.dest.ty.shape))
            reads.clear()
            value = _number(statement.value, bindings, load)
            key = cell(statement.dest, indices)
            memory[key] = value
            if local(statement.dest):
                writes.append((key[1], indices, value))
            elif statement.dest.same_as(output):
                fragment_reads = [item for item in reads if local(item[0])]
                assert len(fragment_reads) == 1
                _, owner, slot = fragment_reads[0]
                output_trace.append((indices, owner, slot, value))
            else:
                raise AssertionError(("unexpected write", statement))
        elif isinstance(statement, ir.Evaluate):
            if isinstance(statement.value, ir.Call):
                assert statement.value.op.name == "tl.annotation"
                fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(
                    statement.value.args[0], statement.value.attrs["annotation_schema"]
                )
                assert set(fields) == {"layout_map"}
            else:
                assert isinstance(statement.value, ir.IntImm)
        else:
            raise AssertionError((type(statement), statement))

    visit(function.body)
    return writes, output_trace


@pytest.mark.parametrize("kind", ["webgpu", "cuda", "metal"])
@pytest.mark.parametrize(
    "shape,layout_style",
    [
        ((1,), "inferred"),
        ((7,), "inferred"),
        ((127,), "inferred"),
        ((128,), "inferred"),
        ((129,), "inferred"),
        ((257,), "inferred"),
        ((3, 43), "inferred"),
        ((257,), "offset-owners"),
    ],
)
def test_fragment_owner_slot_and_guard_match_full_finite_domain(kind, shape, layout_style, tmp_path):
    source, output = tensor("A", shape, "float32", "global"), tensor("B", shape, "float32", "global")
    fragment = tensor("F", shape, "float32", "local.fragment")
    thread = ir.Var("worker", "int32")
    prefix = [allocate(fragment)]
    if layout_style == "offset-owners":
        chosen = Fragment(shape, forward_thread_fn=lambda i: i % 64 + 64, forward_index_fn=lambda i: i // 64)
        prefix.append(annotation({"layout_map": {fragment: chosen}}))
    value = _function(
        [source, output],
        [
            *prefix,
            tileop("fill", region(fragment), ir.FloatImm("float32", 3.5)),
            tileop("copy", region(source), region(fragment)),
            tileop("copy", region(fragment), region(output)),
        ],
        thread=thread,
    ).with_attr("target", target(kind))
    # The public passes must obtain their target from the Function itself.
    module = ir.IRModule({"main": value})
    inferred = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)["main"]
    layouts = [layout for fields in annotations(inferred) for key, layout in fields.get("layout_map", {}).items() if key.same_as(fragment)]
    assert len(layouts) == 1
    layout = layouts[0]
    variables = list(layout.get_forward_vars())
    replicas = int(_number(layout.replicate_size, {}))
    assert len(variables) == len(shape) + (replicas > 1)
    expected, cells = {}, set()
    for indices in product(*(range(size) for size in shape)):
        owners = []
        for replica in range(replicas):
            coordinates = (replica, *indices) if replicas > 1 else indices
            bindings = dict(zip(variables, coordinates))
            owner = int(_number(layout.thread, bindings))
            slot = tuple(int(_number(index, bindings)) for index in layout.index)
            assert (owner, slot) not in cells
            cells.add((owner, slot))
            owners.append((owner, slot))
        expected[indices] = owners

    lowered = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(ir.IRModule({"main": inferred}))["main"]
    assert tirx.analysis.check_well_formed(lowered)
    assert not tirx.analysis.undefined_vars(lowered)
    restored = ir.load_json(ir.save_json(lowered))
    ir.assert_structural_equal(lowered, restored)
    writes, trace = _enumerate_assignments(lowered, source, output, thread, shape)
    assert Counter((owner, slot) for owner, slot, value in writes if value == 3.5) == Counter(cells)
    assert Counter((owner, slot) for owner, slot, value in writes if value != 3.5) == Counter(cells)
    assert len(trace) == int(np.prod(shape))
    assert {indices for indices, _, _, _ in trace} == set(expected)
    assert all((owner, slot) in expected[indices] for indices, owner, slot, _ in trace)
    assert {indices: value for indices, _, _, value in trace} == {
        indices: ordinal + 10 for ordinal, indices in enumerate(product(*(range(size) for size in shape)))
    }
    (tmp_path / "owner-slot-oracle.json").write_text(
        json.dumps(
            {"shape": shape, "target": kind, "expected": [[indices, *mapping] for indices, mapping in expected.items()], "actual": trace},
            indent=2,
        )
    )
    (tmp_path / "lowered.json").write_text(ir.save_json(lowered))

    # LLVM runs these per-thread operations with serialized launch dimensions.
    # This validates values and does not claim execution on the named GPU.
    finalized = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")(kind)(ir.IRModule({"main": lowered}))["main"]
    serial = _serialize_launches(finalized)
    executable = tirx.compile(ir.IRModule({"main": serial}), target=serial.attrs["target"])
    source_array = np.arange(np.prod(shape), dtype="float32").reshape(shape) + 10
    runtime_source = tvm_tirx.runtime.tensor(source_array)
    runtime_output = tvm_tirx.runtime.tensor(np.full(shape, -333, dtype="float32"))
    executable(runtime_source, runtime_output)
    np.testing.assert_array_equal(runtime_output.numpy(), source_array)
    np.testing.assert_array_equal(runtime_source.numpy(), source_array)


@pytest.mark.parametrize("kind", ["webgpu", "cuda", "metal"])
@pytest.mark.parametrize("case", ["different-owners", "colliding-slots", "missing-replicas", "unsupported-slot-stride"])
def test_ragged_fragment_compatibility_does_not_accept_invalid_ownership(kind, case):
    source, destination = tensor("F", [257], "float32", "local.fragment"), tensor("G", [257], "float32", "local.fragment")
    source_layout = Fragment([257], forward_thread_fn=lambda i: i % 128, forward_index_fn=lambda i: i // 128)
    if case == "different-owners":
        destination_layout = Fragment([257], forward_thread_fn=lambda i: (i + 1) % 128, forward_index_fn=lambda i: i // 128)
    elif case == "colliding-slots":
        destination_layout = Fragment([257], forward_thread_fn=lambda i: i % 128, forward_index_fn=lambda i: 0)
    elif case == "missing-replicas":
        destination_layout = Fragment(
            [257], forward_thread_fn=lambda i, replica: i % 64 + replica * 64, forward_index_fn=lambda i: i // 64, replicate=2
        )
    else:
        # The native affine inverse does not support a strided local-slot
        # digit. A padding guard must not turn that missing inverse into
        # silent duplicate accesses or compact the requested storage layout.
        source_layout = Fragment([257], forward_thread_fn=lambda i: i % 128, forward_index_fn=lambda i: (i // 128) * 2)
        destination_layout = source_layout
    value = _function(
        [],
        [
            allocate(source),
            allocate(destination),
            annotation({"layout_map": {source: source_layout, destination: destination_layout}}),
            tileop("fill", region(source), ir.FloatImm("float32", 3.5)),
            tileop("copy", region(source), region(destination)),
        ],
    ).with_attr("target", target(kind))
    before = ir.save_json(value)
    diagnostic = "source->scale" if case == "unsupported-slot-stride" else "[Ll]ayout|injectiv|[Ff]ragment"
    with pytest.raises((ValueError, RuntimeError), match=diagnostic):
        inferred = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(ir.IRModule({"main": value}))
        tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(inferred)
    assert ir.save_json(value) == before
