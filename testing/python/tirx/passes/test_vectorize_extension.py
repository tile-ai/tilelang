import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def make_loop(body, parameters=()):
    index = ir.Var("index", "int32")
    function = tirx.Function(
        list(parameters),
        ir.For(index, 0, 4, 1, ir.ForKind.VECTORIZED, body(index)),
        ret_type=ir.TupleType([]),
    )
    return ir.IRModule({"main": function})


def test_opaque_loop_invariant_calls_execute_per_lane():
    module = make_loop(lambda _: ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("effect")], ty="int32")))
    result = _tirx_passes.VectorizeLoop()(module)["main"]
    loop = result.body.seq[0]
    assert isinstance(loop, ir.For)
    assert loop.kind == ir.ForKind.DEFAULT
    assert int(loop.end - loop.begin) == 4
    assert tirx.analysis.check_well_formed(result)


def test_scalarized_binding_keeps_its_uses_in_scope():
    output = ir.Var("output", tirx.TensorType([4], "int32"))

    def body(index):
        value = ir.Var("value", "int32")
        return ir.SeqStmt(
            [
                ir.Bind(value, ir.Call("tirx.call_extern", [ir.StringImm("effect"), index], ty="int32")),
                ir.TensorStore(output, [index], value + 1),
            ]
        )

    result = _tirx_passes.VectorizeLoop()(make_loop(body, [output]))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert result.body.seq[0].kind == ir.ForKind.DEFAULT


def test_private_tensor_allocation_and_accesses_share_the_scalarized_scope():
    output = ir.Var("output", tirx.TensorType([4], "int32"))

    def body(index):
        tensor_type = tirx.TensorType([1], "int32", scope="local")
        temporary = ir.Var("temporary", tensor_type)
        return ir.SeqStmt(
            [
                ir.Bind(temporary, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor_type])),
                ir.TensorStore(temporary, [0], index * 3 + 2),
                ir.TensorStore(output, [index], ir.TensorLoad(temporary, [0]) + index),
            ]
        )

    result = _tirx_passes.VectorizeLoop()(make_loop(body, [output]))["main"]
    assert tirx.analysis.check_well_formed(result)
    result = result.with_attr("global_symbol", "private_tensor")
    executable = tirx.compile(result, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [2, 6, 10, 14])


def test_pure_lane_assumptions_reduce_to_a_scalar_conjunction():
    extent = ir.Var("extent", "int32")
    module = make_loop(lambda index: ir.Evaluate(ir.Call("prim.assume", [index < extent])), [extent])
    result = _tirx_passes.VectorizeLoop()(module)["main"]
    loops, calls, shuffles = [], [], []
    tvm_ffi.structural_walk(result.body, (ir.For, loops.append))
    tvm_ffi.structural_walk(result.body, (ir.Call, calls.append))
    tvm_ffi.structural_walk(result.body, (ir.prim.Shuffle, shuffles.append))
    assert not loops
    assumption = [call for call in calls if call.op.name == "prim.assume"]
    assert len(assumption) == 1
    assert assumption[0].args[0].ty == ir.PrimType("bool")
    assert sorted(int(shuffle.indices[0]) for shuffle in shuffles) == [0, 1, 2, 3]
    assumption[0].validate()
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("predicate_kind", ["memory", "opaque"])
def test_effectful_assumptions_keep_per_iteration_evaluation(predicate_kind):
    predicates = ir.Var("predicates", tirx.TensorType([4], "bool"))

    def body(index):
        predicate = (
            ir.TensorLoad(predicates, [index])
            if predicate_kind == "memory"
            else ir.Call("tirx.call_extern", [ir.StringImm("check"), index], ty="bool")
        )
        return ir.Evaluate(ir.Call("prim.assume", [predicate]))

    result = _tirx_passes.VectorizeLoop()(make_loop(body, [predicates]))["main"]
    loops = []
    tvm_ffi.structural_walk(result.body, (ir.For, loops.append))
    assert len(loops) == 1
    assert loops[0].kind == ir.ForKind.DEFAULT
    assert int(loops[0].end - loops[0].begin) == 4
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("scalarize", [False, True])
def test_dialect_call_vectorization_extension(scalarize):
    operation = ir.Op.get("prim.exp")
    output = ir.Var("output", tirx.TensorType([4], "float32"))
    seen = []

    def rule(call, variable, lanes, rewrite, target):
        seen.append((call, variable, int(lanes), target))
        if scalarize:
            return None
        return rewrite(call.args[0]) + ir.prim.Broadcast(ir.const(1.0, "float32"), 4)

    operation.set_attr("tl.FVectorizeCall", rule)
    try:
        module = make_loop(
            lambda index: ir.TensorStore(output, [index], ir.Call(operation, [ir.prim.Cast("float32", index)])),
            [output],
        )
        result = _tirx_passes.VectorizeLoop()(module)["main"]
        assert len(seen) == 1
        assert seen[0][2] == 4
        assert tirx.analysis.check_well_formed(result)
        if scalarize:
            assert result.body.seq[0].kind == ir.ForKind.DEFAULT
        else:
            assert isinstance(result.body.seq[0], ir.TensorStore)
            assert result.body.seq[0].value.ty == ir.PrimType("float32x4")
    finally:
        operation.reset_attr("tl.FVectorizeCall")
