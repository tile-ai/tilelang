"""Barriers must cover every reachable machine-integer loop iteration."""

import pytest
from tvm_tirx import ir, tirx

import tilelang
from tilelang.testing.ir import collect_calls


def _kernel(begin, end, step, dtype):
    shared = ir.Var("shared", tirx.TensorType((32,), "float32", scope="shared"))
    output = ir.Var("output", tirx.TensorType((32,), "float32"))
    thread = ir.Var("thread", "int32")
    index = ir.Var("index", dtype)
    negative = index < ir.const(0, dtype)
    # The first iteration is thread-private. On a negative iteration, each
    # thread reads a neighbour's store and therefore requires a barrier.
    shift = ir.prim.Select(negative, ir.const(1), ir.const(0))
    loop = ir.For(
        index,
        ir.const(begin, dtype),
        ir.const(end, dtype),
        ir.const(step, dtype),
        ir.ForKind.DEFAULT,
        [
            ir.TensorStore(shared, [thread], ir.const(1.0)),
            ir.TensorStore(output, [thread], ir.TensorLoad(shared, [(thread + shift) % 32])),
            ir.If(negative, [ir.Break()], None),
        ],
    )
    return tirx.Function(
        [output],
        [
            ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])),
            ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], {}, [loop]),
        ],
    ).with_attr("global_symbol", "main")


@pytest.mark.parametrize(
    "begin,end,step,dtype,visited",
    [
        (0, 2, -1, "int32", [0, -1]),
        (120, 121, 127, "int8", [120, -9]),
    ],
)
def test_thread_sync_does_not_drop_reachable_negative_iteration(begin, end, step, dtype, visited):
    # A finite independent enumeration states the original loop predicate,
    # the declared integer width, and the explicit early exit.
    actual = []
    index = begin
    bits = int(dtype[3:])
    while index < end:
        actual.append(index)
        if index < 0:
            break
        assert len(actual) < 8
        index = (index + step) % (1 << bits)
        if index >= 1 << (bits - 1):
            index -= 1 << bits
    assert actual == visited
    assert any(i < 0 for i in actual)
    function = _kernel(begin, end, step, dtype)
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    transformed = tilelang.transform.ThreadSync("shared")(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(transformed)
    barriers = collect_calls(transformed, op="tirx.gpu_storage_sync")
    assert barriers, "A reachable cross-thread RAW hazard was treated as thread-private"
    assert all(call.args[0].value == "shared" for call in barriers)


def test_descending_endpoints_do_not_create_a_python_range():
    function = _kernel(4, 0, -1, "int32")
    transformed = tilelang.transform.ThreadSync("shared")(ir.IRModule({"main": function}))["main"]
    assert not collect_calls(transformed, op="tirx.gpu_storage_sync")
