"""Native vectorization retains typed tuple operands and lexical metadata."""

import pytest
import tilelang  # noqa: F401
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def _integer(value):
    return ir.IntImm("int32", value)


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def _loop(lane, lanes, body):
    return ir.For(lane, _integer(0), _integer(lanes), _integer(1), ir.ForKind.VECTORIZED, body)


def _region(fields, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _vectorize(params, loop, namespace="tl"):
    function = tirx.Function(params, [loop], ir.NoneType()).with_attr("target", tvm_tirx.target.Target("llvm"))
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    result = tvm_ffi.get_global_func(namespace + ".transform.VectorizeLoop")(True)(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("kind", ["direct", "nested", "projection"])
def test_region_tuple_preserves_varying_uniform_and_tensor_fields(lanes, kind):
    lane, limit = ir.Var("lane", "int32"), ir.Var("limit", "int32")
    output = ir.Var("output", tirx.TensorType([16], "int32", scope="global"))
    loc = ir.SourceLoc(ir.SourceName("tuple_metadata.py"), 12, 2, 12, 24)
    expression = lane + 3
    if kind == "nested":
        expression = ir.Tuple([lane, ir.Tuple([lane < limit, limit], loc=loc)], loc=loc)
    elif kind == "projection":
        expression = ir.TupleGetItem(ir.Tuple([lane + 3, limit], loc=loc), 0, loc=loc)
    region = _region(
        {"varying": expression, "uniform": limit, "tensor": output},
        [ir.TensorStore(output, [lane], lane + 3)],
    )
    result = _vectorize([output, limit], _loop(lane, lanes, [region]))
    assert not _nodes(result, ir.For)
    (rewritten,) = _nodes(result, ir.RegionStmt)
    fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(rewritten.args[0], rewritten.attrs["annotation_schema"])
    assert fields["uniform"].same_as(limit)
    assert fields["tensor"].same_as(output)
    if kind == "nested":
        expected = ir.TupleType(
            [
                ir.PrimType(f"int32x{lanes}"),
                ir.TupleType([ir.PrimType(f"boolx{lanes}"), ir.PrimType("int32")]),
            ]
        )
        ir.assert_structural_equal(fields["varying"].ty, expected)
        assert fields["varying"].loc.same_as(loc)
        assert fields["varying"].fields[1].loc.same_as(loc)
    else:
        ir.assert_structural_equal(fields["varying"].ty, ir.PrimType(f"int32x{lanes}"))
        if kind == "projection":
            assert fields["varying"].loc.same_as(loc)
    (store,) = _nodes(result, ir.TensorStore)
    ir.assert_structural_equal(store.value.ty, ir.PrimType(f"int32x{lanes}"))


@pytest.mark.parametrize("namespace", ["tirx", "tl"])
@pytest.mark.parametrize("field", [0, 1])
def test_tuple_bind_and_projection_share_the_updated_field_type(namespace, field):
    lane, uniform = ir.Var("lane", "int32"), ir.Var("uniform", "int32")
    output = ir.Var("output", tirx.TensorType([4], "int32", scope="global"))
    value = ir.Tuple([lane + 7, uniform])
    bound = ir.Var("values", value.ty)
    result = _vectorize(
        [output, uniform],
        _loop(
            lane,
            4,
            [ir.Bind(bound, value), ir.TensorStore(output, [lane], ir.TupleGetItem(bound, field))],
        ),
        namespace,
    )
    if namespace == "tirx":
        # Native admission intentionally excludes tuple Bind. TileLang owns
        # the additional typed-tuple vector construction contract.
        (scalar_loop,) = _nodes(result, ir.For)
        assert scalar_loop.kind == ir.ForKind.DEFAULT
        (binding,) = _nodes(result, ir.Bind)
        ir.assert_structural_equal(binding.var.ty, value.ty)
        return
    assert not _nodes(result, ir.For)
    (binding,) = _nodes(result, ir.Bind)
    ir.assert_structural_equal(binding.var.ty, ir.TupleType([ir.PrimType("int32x4"), ir.PrimType("int32")]))
    (projection,) = _nodes(result, ir.TupleGetItem)
    assert projection.tuple_value.same_as(binding.var)
    ir.assert_structural_equal(projection.ty, ir.PrimType("int32x4" if field == 0 else "int32"))


@pytest.mark.parametrize("empty", [False, True])
def test_uniform_tuple_metadata_keeps_scalar_types(empty):
    lane, uniform = ir.Var("lane", "int32"), ir.Var("uniform", "int32")
    output = ir.Var("output", tirx.TensorType([4], "int32", scope="global"))
    fields = {} if empty else {"uniform": uniform}
    region = _region(fields, [ir.TensorStore(output, [lane], lane)])
    result = _vectorize([output, uniform], _loop(lane, 4, [region]))
    (rewritten,) = _nodes(result, ir.RegionStmt)
    ir.assert_structural_equal(rewritten.args[0].ty, region.args[0].ty)
    assert not _nodes(result, ir.For)


def test_scalar_only_tuple_operand_scalarizes_its_enclosing_region():
    lane = ir.Var("lane", "int32")
    output = ir.Var("output", tirx.TensorType([4], "int32", scope="global"))
    value = ir.Call("tirx.call_extern", [ir.StringImm("scalar_metadata"), lane], ty_args=[ir.PrimType("int32")])
    region = _region({"value": value}, [ir.TensorStore(output, [lane], lane + 1)])
    result = _vectorize([output], _loop(lane, 4, [region]))
    (loop,) = _nodes(result, ir.For)
    assert loop.kind == ir.ForKind.DEFAULT and int(loop.end - loop.begin) == 4
    (rewritten,) = _nodes(result, ir.RegionStmt)
    fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(rewritten.args[0], rewritten.attrs["annotation_schema"])
    assert fields["value"].args[1].same_as(loop.var)
    (store,) = _nodes(result, ir.TensorStore)
    assert store.indices[0].same_as(loop.var)
