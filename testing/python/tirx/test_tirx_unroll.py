import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def unroll(body, params, config=None):
    function = tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "unroll")
    with ir.transform.PassContext(config={"tl.UnrollLoop": config or {}}):
        result = tvm_ffi.get_global_func("tl.transform.UnrollLoop")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def collect(function, node_type):
    nodes = []
    stmt_functor.post_order_visit(function, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


@pytest.mark.parametrize("begin,end,step", [(0, 8, 1), (3, 16, 4), (-7, 1, 3), (0, 0, 2), (8, 3, 2)])
def test_explicit_unroll_uses_trip_count_and_step(begin, end, step):
    output = TensorView.create("output", [16], "int32")
    index = ir.Var("index", "int32")
    body = ir.For(
        index, begin, end, step, ir.ForKind.UNROLLED, output.store((index - begin) // step, index), {"pragma_unroll_explicit": True}
    )
    function = unroll(body, [output.tensor])
    assert not collect(function, ir.For)
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.full(16, -99, dtype="int32"))
    executable(actual)
    expected = np.full(16, -99, dtype="int32")
    values = np.arange(begin, end, step, dtype="int32")
    expected[: len(values)] = values
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_default_preserves_dynamic_and_factor_loops():
    output = TensorView.create("output", [8], "int32")
    dynamic = ir.Var("dynamic", "int32")
    indices = [ir.Var(name, "int32") for name in ["first", "second", "third"]]
    body = [
        ir.For(indices[0], 0, 2, 1, ir.ForKind.UNROLLED, output.store(indices[0], 1), {"pragma_unroll_explicit": True}),
        ir.For(indices[1], 0, dynamic, 1, ir.ForKind.UNROLLED, output.store(indices[1], 2)),
        ir.For(indices[2], 0, 8, 1, ir.ForKind.UNROLLED, output.store(indices[2], 3), {"pragma_unroll_factor": 4}),
    ]
    function = unroll(body, [output.tensor, dynamic])
    loops = collect(function, ir.For)
    assert len(loops) == 2
    assert loops[0].end.same_as(dynamic)
    assert int(loops[1].annotations["pragma_unroll_factor"]) == 4


def test_dynamic_explicit_unroll_is_rejected():
    dynamic = ir.Var("dynamic", "int32")
    index = ir.Var("index", "int32")
    with pytest.raises(ValueError, match="non-constant"):
        unroll(ir.For(index, 0, dynamic, 1, ir.ForKind.UNROLLED, ir.Evaluate(0)), [dynamic], {"explicit_unroll": True})


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_targeted_loop_control_is_not_fully_expanded(control):
    index = ir.Var("index", "int32")
    body = ir.For(index, 0, 3, 1, ir.ForKind.UNROLLED, control())
    assert len(collect(unroll(body, []), ir.For)) == 1
    with pytest.raises(ValueError, match="cannot be fully expanded"):
        unroll(body, [], {"explicit_unroll": True})


def test_nested_break_does_not_target_outer_loop():
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    body = ir.For(outer, 0, 3, 1, ir.ForKind.UNROLLED, ir.For(inner, 0, 4, 1, ir.ForKind.DEFAULT, ir.Break()))
    function = unroll(body, [], {"explicit_unroll": True})
    loops = collect(function, ir.For)
    assert len(loops) == 3
    assert all(len(collect(loop.body, ir.Break)) == 1 for loop in loops)


def test_local_definitions_are_freshened_without_rebinding_external_storage(native_layouts):
    storage = TensorView.create("storage", [4], "int32")
    output = TensorView.create("output", [4], "int32")
    local = TensorView.create("local", [1], "int32", scope="local")
    alias = TensorView.create("alias", [4], "int32")
    index = ir.Var("index", "int32")
    layout = native_layouts.Layout([1], lambda element: element)
    body = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.UNROLLED,
        [
            local.allocate(),
            ir.Evaluate(ir.Call("tl.layout_hint", [local.tensor], attrs={"layout": layout})),
            alias.declare(storage.data),
            local.store(0, alias.load(index) + index),
            output.store(index, local.load(0)),
        ],
    )
    function = unroll(body, [storage.tensor, output.tensor], {"explicit_unroll": True})
    bindings = collect(function, ir.Bind)
    assert len(bindings) == 8
    assert len({binding.var.__chandle__() for binding in bindings}) == 8
    scopes = collect(function, ir.RegionStmt)
    for scope in scopes:
        bindings_in_scope = collect(scope, ir.Bind)
        hints = [call for call in collect(scope, ir.Call) if call.op.name == "tl.layout_hint"]
        assert len(hints) == 1
        assert hints[0].args[0].same_as(bindings_in_scope[0].var)
    function = stmt_functor.ir_transform(
        function,
        None,
        lambda node: ir.Evaluate(0)
        if isinstance(node, ir.Evaluate) and isinstance(node.value, ir.Call) and node.value.op.name == "tl.layout_hint"
        else None,
    )
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((4,), "int32")
    executable(runtime.tensor(np.arange(4, dtype="int32") * 3), actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(4) * 4)


def test_local_access_and_lexical_policy_do_not_leak():
    local = TensorView.create("local", [4], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    indices = [ir.Var(name, "int32") for name in ["local_index", "scoped_index", "outside_index"]]
    local_loop = ir.For(indices[0], 0, 4, 1, ir.ForKind.DEFAULT, local.store(indices[0], 3))
    scoped_loop = ir.For(indices[1], 0, 4, 1, ir.ForKind.DEFAULT, output.store(indices[1], 4))
    outside_loop = ir.For(indices[2], 0, 4, 1, ir.ForKind.DEFAULT, output.store(indices[2], 5))
    body = [
        local.allocate(),
        local_loop,
        ir.RegionStmt("tirx.scope", [], [], {"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": True}, scoped_loop),
        outside_loop,
    ]
    function = unroll(body, [output.tensor], {"unroll_local_access": True})
    loops = collect(function, ir.For)
    assert len(loops) == 2
    assert loops[0].kind == ir.ForKind.UNROLLED
    assert loops[1].kind == ir.ForKind.DEFAULT
