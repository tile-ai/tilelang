"""Lexical thread origins are proved locally and preserve exact event ownership."""

from collections import Counter
import json
import operator

import pytest
from tvm_ffi.testing import create_object
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.layout import Fragment


def integer(value):
    return ir.const(value, "int32")


def _function(mode="serial", thread_begin=0, padded=False, shifted_slot=False, thread_layout="adjacent"):
    rows, columns, lanes = 3, 16, 8
    a = ir.Var("row", "int32")
    b = ir.Var("origin" if mode == "unknown" else "row", "int32")
    thread, index = ir.Var("thread", "int32"), ir.Var("k", "int32")
    params = []
    if mode == "nested":
        rows = 4
        logical_row = a * 2 + b
        origin = logical_row * lanes
    elif mode in ("unknown", "shadowed_unknown"):
        params.append(b)
        logical_row, origin = a, b * lanes
    elif mode == "overflow":
        logical_row, origin = a, integer(2**31 - 4)
    else:
        logical_row, origin = a, a * lanes
    extent = rows * lanes if mode != "overflow" else 2**31 - 1
    total_threads = thread_begin + extent
    output = ir.Var("events", tirx.TensorType([rows, columns, total_threads], "int32", scope="global"))
    params.insert(0, output)
    owner = (lambda k: k // 2) if thread_layout == "adjacent" else (lambda k: k % 4 + (k // 8) * 4)
    slot = (lambda k: k % 2) if thread_layout == "adjacent" else (lambda k: (k // 4) % 2)
    layout = Fragment((columns,), forward_fn=lambda k: (origin + owner(k), slot(k) + (1 if shifted_slot else 0)))
    layout = create_object(
        "tl.Fragment",
        input_size=layout.input_size,
        forward_index=layout.forward_index,
        forward_thread=layout.forward_thread,
        replicate_size=layout.replicate_size,
        thread_range=ir.Range(integer(thread_begin), integer(thread_begin + extent)),
    )
    indices = [logical_row, index, thread]
    store = ir.TensorStore(output, indices, ir.TensorLoad(output, indices) + 1)
    body = ir.For(
        index,
        integer(0),
        integer(columns),
        integer(1),
        ir.ForKind.PARALLEL,
        [store],
        annotations={"parallel_loop_layout": layout, "parallel_loop_requires_padding_guard": padded},
    )
    if mode == "nested":
        body = ir.For(b, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [body])
        body = ir.For(a, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [body])
    else:
        if mode == "guarded":
            body = ir.If((a >= 0) & (a < rows), [body], None)
        begin = -1 if mode in ("negative", "guarded") else 0
        end = rows + 1 if mode in ("out_of_range", "guarded") else rows
        body = ir.For(a, integer(begin), integer(end), integer(1), ir.ForKind.DEFAULT, [body])
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(total_threads)], [thread], {}, [body])
    target = Target({"kind": "cuda", "arch": "sm_90"})
    function = tirx.Function(params, [launch], ir.NoneType(), ir.DictAttrs({"global_symbol": "main", "target": target}))
    expected = Counter(
        (row, k, thread_begin + row * lanes + (k // 2 if thread_layout == "adjacent" else k % 4 + (k // 8) * 4))
        for row in range(rows)
        for k in range(columns)
    )
    return function, layout, expected


def _lower(function, tmp_path):
    before = ir.save_json(function)
    assert tirx.analysis.check_well_formed(function)
    (tmp_path / "input.json").write_text(before)
    try:
        with function.attrs["target"]:
            lowered = tilelang.transform.LowerTileOp()(ir.IRModule({"main": function}))["main"]
    finally:
        assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(lowered)
    assert not tirx.analysis.undefined_vars(lowered)
    (tmp_path / "lowered.json").write_text(ir.save_json(lowered))
    (tmp_path / "lowered.txt").write_text(str(lowered))
    return lowered


def _integer(expr, values):
    if isinstance(expr, ir.IntImm):
        return int(expr)
    if isinstance(expr, ir.Var):
        return values[expr]
    operations = {
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
        "And": lambda a, b: a and b,
        "Or": lambda a, b: a or b,
        "BitwiseAnd": operator.and_,
        "BitwiseOr": operator.or_,
        "BitwiseXor": operator.xor,
    }
    name = type(expr).__name__
    if name in operations:
        return operations[name](_integer(expr.a, values), _integer(expr.b, values))
    if name == "Not":
        return not _integer(expr.a, values)
    if name == "Cast":
        value = _integer(expr.value, values)
        assert -(2**31) <= value < 2**31
        return value
    raise AssertionError((type(expr), str(expr)))


def _events(function):
    """Enumerate lowered stores without consulting symbolic simplification."""
    events = Counter()

    def visit(node, values):
        if isinstance(node, ir.SeqStmt):
            for child in node.seq:
                visit(child, values)
        elif isinstance(node, ir.RegionStmt):
            assert node.op.name == "tirx.launch_thread"
            (var,) = node.body_params
            for tid in range(_integer(node.args[1], values)):
                visit(node.body, {**values, var: tid})
        elif isinstance(node, ir.For):
            begin, end, step = (_integer(x, values) for x in (node.begin, node.end, node.step))
            assert 0 <= end - begin <= 64
            for value in range(begin, end, step):
                visit(node.body, {**values, node.var: value})
        elif isinstance(node, ir.If):
            branch = node.then_case if _integer(node.condition, values) else node.else_case
            if branch is not None:
                visit(branch, dict(values))
        elif isinstance(node, ir.Bind):
            values[node.var] = _integer(node.value, values)
        elif isinstance(node, ir.TensorStore):
            events[tuple(_integer(x, values) for x in node.indices)] += 1
        else:
            raise AssertionError((type(node), str(node)))

    visit(function.body, {})
    return events


@pytest.mark.parametrize("mode", ["serial", "nested", "guarded"])
@pytest.mark.parametrize("thread_begin", [0, 8])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("thread_layout", ["adjacent", "interleaved"])
def test_lexical_origin_preserves_every_owner_and_event(mode, thread_begin, padded, thread_layout, tmp_path):
    function, layout, expected = _function(mode, thread_begin, padded, thread_layout=thread_layout)
    before = ir.save_json(layout)
    lowered = _lower(function, tmp_path)
    assert ir.save_json(layout) == before
    events = _events(lowered)
    (tmp_path / "events.json").write_text(json.dumps({"actual": sorted(events.items()), "expected": sorted(expected.items())}) + "\n")
    assert events == expected


@pytest.mark.parametrize("mode", ["unknown", "shadowed_unknown", "negative", "out_of_range", "overflow"])
def test_unproved_or_invalid_origin_is_rejected(mode, tmp_path):
    function, _, _ = _function(mode)
    with pytest.raises(ValueError, match="Cannot infer a safe output extent for layout coordinate") as error:
        _lower(function, tmp_path)
    (tmp_path / "expected-error.txt").write_text(str(error.value))


def test_thread_origin_with_padded_slot_keeps_guarded_inverse(tmp_path):
    function, _, expected = _function("serial", 8, True, shifted_slot=True)
    assert _events(_lower(function, tmp_path)) == expected
