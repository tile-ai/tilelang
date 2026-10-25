from tilelang import tvm
from tilelang.ascend import transform
from tilelang._compat import tirx
from tvm_tirx import ir


def _restore(body, loop_var, output=None):
    if output is None:
        output = body.dest
    synthetic_loop = ir.For(
        loop_var,
        3,
        11,
        1,
        ir.ForKind.DEFAULT,
        body,
        annotations={"synthetic_while": 1},
    )
    func = tirx.PrimFunc(
        [output],
        synthetic_loop,
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "main")
    return transform.RestoreWhileLoops()(tvm.IRModule.from_expr(func))["main"].body


def test_restore_while_loops_preserves_residual_loop_var_with_counter():
    output = tirx.decl_buffer((16,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    body = _restore(
        ir.TensorStore(output, [loop_var], loop_var),
        loop_var,
    )

    assert isinstance(body, ir.SeqStmt)
    allocation, initialize, loop = body.seq
    assert isinstance(allocation, ir.Bind)
    assert allocation.value.op.name == "tirx.alloc_tensor"
    assert isinstance(initialize, ir.TensorStore)
    assert initialize.value.value == 3
    assert isinstance(loop, ir.While)

    restored_store, increment = loop.body.seq
    assert isinstance(restored_store.value, ir.TensorLoad)
    assert restored_store.value.source.same_as(allocation.var)
    assert isinstance(restored_store.indices[0], ir.TensorLoad)
    assert restored_store.indices[0].source.same_as(allocation.var)
    assert isinstance(increment, ir.TensorStore)
    assert increment.dest.same_as(allocation.var)


def test_restore_while_loops_avoids_counter_when_loop_var_is_dead():
    output = tirx.decl_buffer((1,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    body = _restore(
        ir.TensorStore(output, [0], 7),
        loop_var,
    )

    (loop,) = body.seq
    assert isinstance(loop, ir.While)
    assert isinstance(loop.body.seq[0], ir.TensorStore)


def test_restore_while_loops_avoids_counter_after_version_index_simplification():
    output = tirx.decl_buffer((2,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    inner_var = ir.Var("inner", "int32")
    version_index = (loop_var * 4 + inner_var) % 2
    scheduled_body = ir.For(
        inner_var,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        ir.TensorStore(output, [version_index], 7),
    )

    body = _restore(scheduled_body, loop_var, output)

    (loop,) = body.seq
    assert isinstance(loop, ir.While)
    (inner,) = loop.body.seq
    assert isinstance(inner, ir.For)
    restored_store = inner.body.seq[0]
    assert isinstance(restored_store, ir.TensorStore)
    tvm.ir.assert_structural_equal(restored_store.indices[0], inner_var % 2)


if __name__ == "__main__":
    test_restore_while_loops_preserves_residual_loop_var_with_counter()
    test_restore_while_loops_avoids_counter_when_loop_var_is_dead()
    test_restore_while_loops_avoids_counter_after_version_index_simplification()
