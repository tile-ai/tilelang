"""User copy policies survive inferred layout wrappers and private replanning."""

import pytest
import tilelang
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def _integer(value):
    return ir.IntImm("int32", value)


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def _annotations(fields, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "tl.loop.annotations", "value": fields})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


@pytest.mark.parametrize("native_literal", [False, True])
@pytest.mark.parametrize(
    "outer,inner,expected_async", [(True, None, True), (True, False, False), (False, True, True), (False, None, False)]
)
def test_nested_loop_hints_compose_with_inner_priority_and_no_sibling_leak(outer, inner, expected_async, native_literal):
    source, output = [ir.Var(name, tirx.TensorType([16], "float32", scope="global")) for name in ("A", "B")]
    first, second = [ir.Var(name, tirx.TensorType([16], "float32", scope="shared")) for name in ("S0", "S1")]
    thread, block = ir.Var("thread", "int32"), ir.Var("block", "int32")
    key = "parallel_async_without_async_commit_wait"

    def flag(value):
        return ir.IntImm("bool", value) if native_literal else value

    def copy(dest):
        lane = ir.Var("lane", "int32")
        return ir.For(
            lane,
            _integer(0),
            _integer(16),
            _integer(1),
            ir.ForKind.PARALLEL,
            [ir.TensorStore(dest, [lane], ir.TensorLoad(source, [lane]))],
        )

    fields = {"test_tag": "inner"}
    if inner is not None:
        fields[key] = flag(inner)
    first_copy = _annotations({key: flag(outer)}, [_annotations(fields, [copy(first)])])
    body = [
        ir.Bind(first, ir.Call("tirx.alloc_tensor", [], ty_args=[first.ty])),
        ir.Bind(second, ir.Call("tirx.alloc_tensor", [], ty_args=[second.ty])),
        first_copy,
        copy(second),
        ir.TensorStore(output, [thread], ir.TensorLoad(first, [thread]) + ir.TensorLoad(second, [thread])),
    ]
    launch = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("blockIdx.x"), _integer(1)],
        [block],
        {},
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), _integer(16)], [thread], {}, body)],
    )
    function = tirx.Function([source, output], [launch], ir.NoneType()).with_attr(
        "target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
    )
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    inferred = tilelang.transform.LayoutInference()(ir.IRModule({"main": function}))
    lowered = tilelang.transform.LowerTileOp()(inferred)["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(lowered)
    assert not tirx.analysis.undefined_vars(lowered)
    calls = _nodes(lowered, ir.Call)
    copies = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tl.ptx_cp_async"]
    assert len(copies) == int(expected_async)
    if copies:
        assert copies[0].args[0].args[0].source.name == "S0"
    assert all(call.op.name not in ("tl.ptx_commit_group", "tl.ptx_wait_group") for call in calls if isinstance(call.op, ir.Op))
    assert any(store.dest.name == "S1" for store in _nodes(lowered, ir.TensorStore))
