from tilelang import tvm
from tilelang.ascend import transform


def _restore(body, loop_var, output=None):
    if output is None:
        output = body.buffer
    synthetic_loop = tvm.tirx.For(
        loop_var,
        3,
        8,
        tvm.tirx.ForKind.SERIAL,
        body,
        annotations={"synthetic_while": 1},
    )
    func = tvm.tirx.PrimFunc(
        [output.data],
        synthetic_loop,
        buffer_map={output.data: output},
    ).with_attr("global_symbol", "main")
    return transform.RestoreWhileLoops()(tvm.IRModule.from_expr(func))["main"].body


def test_restore_while_loops_preserves_residual_loop_var_with_counter():
    output = tvm.tirx.decl_buffer((16,), "int32", name="output")
    loop_var = tvm.tirx.Var("synthetic", "int32")
    body = _restore(
        tvm.tirx.BufferStore(output, loop_var, [loop_var]),
        loop_var,
    )

    assert isinstance(body, tvm.tirx.SeqStmt)
    allocation, initialize, loop = body.seq
    assert isinstance(allocation, tvm.tirx.AllocBuffer)
    assert isinstance(initialize, tvm.tirx.BufferStore)
    assert initialize.value.value == 3
    assert isinstance(loop, tvm.tirx.While)

    restored_store, increment = loop.body.seq
    assert isinstance(restored_store.value, tvm.tirx.BufferLoad)
    assert restored_store.value.buffer.same_as(allocation.buffer)
    assert isinstance(restored_store.indices[0], tvm.tirx.BufferLoad)
    assert restored_store.indices[0].buffer.same_as(allocation.buffer)
    assert isinstance(increment, tvm.tirx.BufferStore)
    assert increment.buffer.same_as(allocation.buffer)


def test_restore_while_loops_avoids_counter_when_loop_var_is_dead():
    output = tvm.tirx.decl_buffer((1,), "int32", name="output")
    loop_var = tvm.tirx.Var("synthetic", "int32")
    body = _restore(
        tvm.tirx.BufferStore(output, 7, [0]),
        loop_var,
    )

    assert isinstance(body, tvm.tirx.While)
    assert isinstance(body.body, tvm.tirx.BufferStore)


def test_restore_while_loops_avoids_counter_after_version_index_simplification():
    output = tvm.tirx.decl_buffer((2,), "int32", name="output")
    loop_var = tvm.tirx.Var("synthetic", "int32")
    inner_var = tvm.tirx.Var("inner", "int32")
    version_index = tvm.tirx.indexmod(loop_var * 4 + inner_var, 2)
    scheduled_body = tvm.tirx.For(
        inner_var,
        0,
        4,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.BufferStore(output, 7, [version_index]),
    )

    body = _restore(scheduled_body, loop_var, output)

    assert isinstance(body, tvm.tirx.While)
    assert isinstance(body.body, tvm.tirx.For)
    restored_store = body.body.body
    assert isinstance(restored_store, tvm.tirx.BufferStore)
    tvm.ir.assert_structural_equal(restored_store.indices[0], tvm.tirx.indexmod(inner_var, 2))


if __name__ == "__main__":
    test_restore_while_loops_preserves_residual_loop_var_with_counter()
    test_restore_while_loops_avoids_counter_when_loop_var_is_dead()
    test_restore_while_loops_avoids_counter_after_version_index_simplification()
