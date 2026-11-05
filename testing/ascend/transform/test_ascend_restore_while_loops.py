from tvm_tirx import tirx

from tvm_tirx import ir
from testing.ascend._ir import access_tensor, single_statement, tensor
import tvm_tirx as tvm
from tilelang.ascend import transform


def _restore(body, loop_var, output=None):
    if output is None:
        output = access_tensor(body)
    synthetic_loop = ir.For(
        loop_var, ir.const(3), ir.const(3) + ir.const(8), ir.const(1), ir.ForKind.DEFAULT, body, annotations={"synthetic_while": 1}
    )
    func = tirx.Function([output], synthetic_loop).with_attr("global_symbol", "main")
    return single_statement(transform.RestoreWhileLoops()(tvm.IRModule.from_expr(func))["main"].body)


def test_restore_while_loops_preserves_residual_loop_var_with_counter():
    output = tensor((16,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    body = _restore(
        ir.TensorStore(output, [loop_var], loop_var),
        loop_var,
    )

    assert isinstance(body, ir.SeqStmt)
    allocation, initialize, loop = body.seq
    assert isinstance(allocation, ir.Bind) and allocation.value.op.name == "tirx.alloc_tensor"
    assert isinstance(initialize, ir.TensorStore)
    assert initialize.value.value == 3
    assert isinstance(loop, ir.While)

    snapshot, increment, restored_store = loop.body.seq
    # Native SSA captures the iteration value before incrementing the mutable
    # counter. All original uses see that same value throughout this iteration.
    assert isinstance(snapshot, ir.Bind) and isinstance(snapshot.value, ir.TensorLoad)
    assert access_tensor(snapshot.value).same_as(allocation.var)
    assert restored_store.value.same_as(snapshot.var)
    assert restored_store.indices[0].same_as(snapshot.var)
    assert isinstance(increment, ir.TensorStore)
    assert access_tensor(increment).same_as(allocation.var)
    tvm.ir.assert_structural_equal(increment.value, snapshot.var + 1)


def test_restore_while_loops_avoids_counter_when_loop_var_is_dead():
    output = tensor((1,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    body = _restore(
        ir.TensorStore(output, [0], 7),
        loop_var,
    )

    assert isinstance(body, ir.While)
    assert isinstance(single_statement(body.body), ir.TensorStore)


def test_restore_while_loops_avoids_counter_after_version_index_simplification():
    output = tensor((2,), "int32", name="output")
    loop_var = ir.Var("synthetic", "int32")
    inner_var = ir.Var("inner", "int32")
    version_index = (loop_var * 4 + inner_var) % 2
    scheduled_body = ir.For(
        inner_var, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, ir.TensorStore(output, [version_index], 7)
    )

    body = _restore(scheduled_body, loop_var, output)

    assert isinstance(body, ir.While)
    assert isinstance(single_statement(body.body), ir.For)
    restored_store = single_statement(single_statement(body.body).body)
    assert isinstance(restored_store, ir.TensorStore)
    tvm.ir.assert_structural_equal(restored_store.indices[0], inner_var % 2)


if __name__ == "__main__":
    test_restore_while_loops_preserves_residual_loop_var_with_counter()
    test_restore_while_loops_avoids_counter_when_loop_var_is_dead()
    test_restore_while_loops_avoids_counter_after_version_index_simplification()
