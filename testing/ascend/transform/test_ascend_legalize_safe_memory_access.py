import tilelang as tl
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _collect_call_nodes(stmt, op_name):
    calls = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == op_name:
            calls.append(node)

    post_order_visit(stmt, _visit)
    return calls


def _is_call_to(expr, op_name):
    return isinstance(expr, tvm.tirx.Call) and isinstance(expr.op, tvm.ir.Op) and str(expr.op.name) == op_name


def _is_int_zero(expr):
    return isinstance(expr, tvm.tirx.IntImm) and int(expr.value) == 0


def test_dynamic_index_load_store_conditions_are_combined():
    @T.prim_func
    def main(
        A: T.Tensor((16,), T.float32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[index]
        A[index] = T.float32(1)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = _collect_call_nodes(body, "tirx.if_then_else")
    assert len(load_guards) == 1
    assert isinstance(load_guards[0].args[0], tvm.tirx.And)
    assert isinstance(load_guards[0].args[1], tvm.tirx.BufferLoad)
    assert isinstance(load_guards[0].args[2], tvm.tirx.FloatImm)
    assert float(load_guards[0].args[2].value) == 0.0

    store_guards = []

    def _collect_store_guard(node):
        if isinstance(node, tvm.tirx.IfThenElse):
            store_guards.append(node)

    post_order_visit(body, _collect_store_guard)
    assert len(store_guards) == 1
    assert isinstance(store_guards[0].condition, tvm.tirx.And)
    assert isinstance(store_guards[0].then_case, tvm.tirx.BufferStore)
    tvm.ir.assert_structural_equal(load_guards[0].args[0], store_guards[0].condition)


def test_nested_buffer_load_index_is_safely_rewritten():
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((4,), T.int32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[B[index]]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data
    b_data = main.buffer_map[main.params[1]].data

    def _is_load_from(expr, buffer_data):
        return isinstance(expr, tvm.tirx.BufferLoad) and expr.buffer.data.same_as(buffer_data)

    def _assert_safe_b_index(expr):
        assert _is_call_to(expr, "tirx.if_then_else")
        assert isinstance(expr.args[0], tvm.tirx.And)
        assert _is_load_from(expr.args[1], b_data)
        assert _is_int_zero(expr.args[2])

    all_guards = _collect_call_nodes(body, "tirx.if_then_else")
    inner_guards = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], a_data)]
    assert len(inner_guards) == 1

    inner_guard = inner_guards[0]
    outer_guards = [call for call in all_guards if len(call.args) == 3 and call.args[1].same_as(inner_guard)]
    assert len(outer_guards) == 1

    outer_guard = outer_guards[0]
    for fallback in (inner_guard.args[2], outer_guard.args[2]):
        assert isinstance(fallback, tvm.tirx.FloatImm)
        assert float(fallback.value) == 0.0
    _assert_safe_b_index(inner_guard.args[1].indices[0])

    for comparison in (inner_guard.args[0], outer_guard.args[0]):
        assert isinstance(comparison, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))
        safe_indices = [operand for operand in (comparison.a, comparison.b) if _is_call_to(operand, "tirx.if_then_else")]
        assert len(safe_indices) == 1
        _assert_safe_b_index(safe_indices[0])

    guarded_b_indices = [
        call for call in _collect_call_nodes(body, "tirx.if_then_else") if len(call.args) == 3 and _is_load_from(call.args[1], b_data)
    ]
    assert len(guarded_b_indices) == 1


def test_ramp_load_conditions_are_combined():
    lanes = 4

    @T.prim_func
    def main(
        A: T.Tensor((16,), T.float32),
        out: T.Tensor((lanes,), T.float32),
        base: T.int32,
    ):
        out[T.Ramp(0, 1, lanes)] = A[T.Ramp(base, 1, lanes)]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = [
        call
        for call in _collect_call_nodes(body, "tirx.if_then_else")
        if len(call.args) == 3 and isinstance(call.args[1], tvm.tirx.BufferLoad)
    ]
    assert len(load_guards) == 1

    guard = load_guards[0]
    assert guard.args[0].dtype.lanes == 1
    assert isinstance(guard.args[0], tvm.tirx.And)
    assert guard.args[1].dtype.lanes == lanes
    assert isinstance(guard.args[2], tvm.tirx.Broadcast)
    assert guard.args[2].dtype.lanes == lanes

    predicate_nodes = []
    post_order_visit(guard.args[0], predicate_nodes.append)
    comparisons = [node for node in predicate_nodes if isinstance(node, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))]
    # Prefix constraints eliminate the three lower-bound checks implied by
    # `0 <= base`, while retaining the four progressively tighter upper bounds.
    assert len(comparisons) == lanes + 1
    assert all(not isinstance(node, tvm.tirx.BufferLoad) for node in predicate_nodes)
    assert all(node.dtype.lanes == 1 for node in predicate_nodes if hasattr(node, "dtype"))
    assert len(_collect_call_nodes(guard.args[0], "tirx.if_then_else")) == 0


def _collect_nodes(stmt, node_type):
    nodes = []

    def _visit(node):
        if isinstance(node, node_type):
            nodes.append(node)

    post_order_visit(stmt, _visit)
    return nodes


def _comparison_uses_var(comparison, var):
    assert isinstance(comparison, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))
    return comparison.a.same_as(var) or comparison.b.same_as(var)


def _assert_single_opaque_call(body):
    calls = _collect_call_nodes(body, "tirx.call_extern")
    assert len(calls) == 1
    assert all(str(call.args[0].value) == "tl_test_opaque_get_index" for call in calls)


def _assert_expr_guard_scope(body, index_var):
    guards = [call for call in _collect_call_nodes(body, "tirx.if_then_else") if len(call.args) == 3]
    assert len(guards) == 4
    assert all(not isinstance(guard.args[0], tvm.tirx.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = outer_lower.args[1]
    assert _comparison_uses_var(outer_lower.args[0], index_var)
    assert _comparison_uses_var(outer_upper.args[0], index_var)

    inner_lower = outer_upper.args[1]
    inner_upper = inner_lower.args[1]
    assert _is_call_to(inner_lower, "tirx.if_then_else")
    assert _is_call_to(inner_upper, "tirx.if_then_else")
    assert not _comparison_uses_var(inner_lower.args[0], index_var)
    assert not _comparison_uses_var(inner_upper.args[0], index_var)
    assert isinstance(inner_upper.args[1], tvm.tirx.BufferLoad)
    return inner_upper.args[1]


def _assert_stmt_guard_scope(body, index_var):
    guards = [guard for guard in _collect_nodes(body, tvm.tirx.IfThenElse)]
    assert len(guards) == 4
    assert all(not isinstance(guard.condition, tvm.tirx.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = outer_lower.then_case
    assert _comparison_uses_var(outer_lower.condition, index_var)
    assert _comparison_uses_var(outer_upper.condition, index_var)

    inner_lower = outer_upper.then_case
    inner_upper = inner_lower.then_case
    assert isinstance(inner_lower, tvm.tirx.IfThenElse)
    assert isinstance(inner_upper, tvm.tirx.IfThenElse)
    assert not _comparison_uses_var(inner_lower.condition, index_var)
    assert not _comparison_uses_var(inner_upper.condition, index_var)
    assert isinstance(inner_upper.then_case, tvm.tirx.BufferStore)
    return inner_upper.then_case


def test_opaque_index_guard_preserves_lazy_evaluation():
    @T.prim_func
    def main(
        A: T.Tensor((4, 4), T.float32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[
            T.call_extern("int32", "tl_test_opaque_get_index") + index,
            index,
        ]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data

    load = _assert_expr_guard_scope(body, main.params[-1])
    assert load.buffer.data.same_as(a_data)
    assert len(_collect_call_nodes(load.indices[0], "tirx.call_extern")) == 1
    _assert_single_opaque_call(body)


def test_opaque_store_guard_preserves_lazy_evaluation():
    @T.prim_func
    def main(
        A: T.Tensor((4, 4), T.float32),
        index: T.int32,
    ):
        A[
            T.call_extern("int32", "tl_test_opaque_get_index") + index,
            index,
        ] = T.float32(1)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data

    store = _assert_stmt_guard_scope(body, main.params[-1])
    assert store.buffer.data.same_as(a_data)
    assert len(_collect_call_nodes(store.indices[0], "tirx.call_extern")) == 1
    _assert_single_opaque_call(body)


def test_producer_load_index_guard_preserves_lazy_evaluation():
    producer = tvm.te.placeholder((4,), dtype="int32", name="P")
    buffer = tvm.tirx.decl_buffer((4, 4), "float32", name="A")
    output = tvm.tirx.decl_buffer((1,), "float32", name="out")
    producer_index = tvm.tirx.Var("producer_index", "int32")
    later_index = tvm.tirx.Var("later_index", "int32")
    mutable_index = tvm.tirx.ProducerLoad(producer, [producer_index])
    load = tvm.tirx.BufferLoad(buffer, [mutable_index, later_index])
    body = tvm.tirx.BufferStore(output, load, [0])
    func = tvm.tirx.PrimFunc(
        [buffer.data, output.data, producer_index, later_index],
        body,
        buffer_map={buffer.data: buffer, output.data: output},
    ).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    guarded = transformed["main"].body.value
    conditions = []
    for _ in range(4):
        assert _is_call_to(guarded, "tirx.if_then_else")
        conditions.append(guarded.args[0])
        guarded = guarded.args[1]

    assert isinstance(guarded, tvm.tirx.BufferLoad)
    assert all(not isinstance(condition, tvm.tirx.And) for condition in conditions)
    assert all(_comparison_uses_var(condition, later_index) for condition in conditions[:2])
    assert all(not _collect_nodes(condition, tvm.tirx.ProducerLoad) for condition in conditions[:2])
    assert all(_collect_nodes(condition, tvm.tirx.ProducerLoad) for condition in conditions[2:])
    assert _collect_nodes(guarded.indices[0], tvm.tirx.ProducerLoad)


if __name__ == "__main__":
    tilelang.testing.main()
