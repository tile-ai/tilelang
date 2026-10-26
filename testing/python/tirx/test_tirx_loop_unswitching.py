import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def unswitch(function, **config):
    with ir.transform.PassContext(config=config):
        result = tvm_ffi.get_global_func("tl.transform.LoopUnswitching")()(ir.IRModule({"main": function}))["main"]
    assert ir.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


@pytest.mark.parametrize("with_else", [False, True])
@pytest.mark.parametrize("allow_else", [False, True])
def test_unswitching_controls_code_duplication(with_else, allow_else, *, compile_tilelang):
    output = TensorView.create("output", [8], "int32")
    condition = ir.Var("condition", "bool")
    index = ir.Var("index", "int32")
    branch = ir.If(condition, output.store(index, index + 1), output.store(index, 19) if with_else else None)
    function = tirx.Function([output.tensor, condition], ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, branch)).with_attr(
        "global_symbol", "unswitch"
    )
    result = unswitch(function, **{"tl.loop_unswitching_allow_non_trivial_else": allow_else})
    first = result.body.seq[0]
    assert isinstance(first, ir.If if allow_else or not with_else else ir.For)
    ir.assert_structural_equal(result, unswitch(result, **{"tl.loop_unswitching_allow_non_trivial_else": allow_else}))
    executable = compile_tilelang(result, target="c")
    for enabled in [False, True]:
        actual = runtime.tensor(np.full(8, 7, dtype="int32"))
        executable(actual, enabled)
        expected = np.arange(1, 9) if enabled else np.full(8, 19 if with_else else 7)
        np.testing.assert_array_equal(actual.numpy(), expected)


def test_unswitching_can_be_disabled():
    output = TensorView.create("output", [8], "int32")
    condition = ir.Var("condition", "bool")
    index = ir.Var("index", "int32")
    function = tirx.Function(
        [output.tensor, condition],
        ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, ir.If(condition, output.store(index, 1), None)),
    )
    ir.assert_structural_equal(function, unswitch(function, **{"tl.disable_loop_unswitching": True}))


@pytest.mark.parametrize("annotated", [False, True])
def test_stable_memory_guard_hoists_inside_region(annotated):
    output = TensorView.create("output", [8], "int32")
    guard = TensorView.create("guard", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, ir.If(guard.load(0) > 0, output.store(index, index), None))
    attributes = {"name_hint": "tilelang_root", "tl.launch_threads": [32]} if annotated else {}
    region = ir.RegionStmt("tl.scope", [], [], ir.DictAttrs(attributes), [guard.allocate(), guard.store(0, 1), loop])
    function = tirx.Function([output.tensor], region)
    result = unswitch(function)
    scope = result.body.seq[0]
    assert isinstance(scope, ir.RegionStmt)
    assert dict(scope.attrs) == dict(region.attrs)
    assert isinstance(scope.body.seq[-1], ir.If)
    assert isinstance(scope.body.seq[-1].then_case.seq[0], ir.For)


def test_mutable_predicate_is_not_hoisted(*, compile_tilelang):
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "int32")
    function = tirx.Function(
        [output.tensor],
        ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, ir.If(output.load(0) < 4, output.store(0, output.load(0) + 1), None)),
    ).with_attr("global_symbol", "mutable_predicate")
    result = unswitch(function)
    assert isinstance(result.body.seq[0], ir.For)
    executable = compile_tilelang(result, target="c")
    actual = runtime.tensor(np.array([0], dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [4])


def test_two_sided_unswitching_preserves_ssa_and_nested_else(*, compile_tilelang):
    output = TensorView.create("output", [8], "int32")
    condition = ir.Var("condition", "bool")
    index = ir.Var("index", "int32")
    local = ir.Var("local", "int32")
    branch = ir.If(index < 4, ir.If(condition, [ir.Bind(local, index + 1), output.store(index, local)], None), output.store(index, 31))
    function = tirx.Function([output.tensor, condition], ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, branch)).with_attr(
        "global_symbol", "nested_else"
    )
    result = unswitch(function, **{"tl.loop_unswitching_allow_non_trivial_else": True})
    definitions = []
    stmt_functor.post_order_visit(result, lambda node: definitions.append(node.var) if isinstance(node, ir.Bind) else None)
    assert len({variable.__hash__() for variable in definitions}) == len(definitions)
    executable = compile_tilelang(result, target="c")
    for enabled in [False, True]:
        actual = runtime.tensor(np.full(8, 7, dtype="int32"))
        executable(actual, enabled)
        expected = np.full(8, 31, dtype="int32")
        expected[:4] = np.arange(1, 5) if enabled else 7
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("noalias", [False, True])
@pytest.mark.parametrize("bound_predicate", [False, True])
@pytest.mark.parametrize("begin,end,step", [(0, 6, 1), (1, 6, 2), (5, -1, -2)])
def test_stable_memory_predicate_respects_alias_contract(noalias, bound_predicate, begin, end, step, *, compile_tilelang):
    output = TensorView.create("output", [6], "int32")
    guard = TensorView.create("guard", [1], "int32")
    index = ir.Var("index", "int32")
    condition = guard.load(0) > 0
    body = []
    if bound_predicate:
        predicate = ir.Var("predicate", "bool")
        body.append(ir.Bind(predicate, condition))
        condition = predicate
    body.append(ir.If(condition, output.store(index, index + 1), None))
    function = tirx.Function([output.tensor, guard.tensor], ir.For(index, begin, end, step, ir.ForKind.DEFAULT, body))
    function = function.with_attr("global_symbol", "stable_guard").with_attr("tirx.noalias", noalias)
    result = unswitch(function)
    assert any(isinstance(statement, ir.If) for statement in result.body.seq) == noalias
    executable = compile_tilelang(result, target="c")
    for enabled in [0, 1]:
        actual = runtime.tensor(np.zeros(6, dtype="int32"))
        executable(actual, runtime.tensor(np.array([enabled], dtype="int32")))
        expected = np.zeros(6, dtype="int32")
        if enabled:
            for position in range(begin, end, step):
                expected[position] = position + 1
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("alias_kind", ["parameter", "view", "nonrestrict"])
def test_aliasing_memory_predicate_remains_inside_loop(alias_kind, *, compile_tilelang):
    output = TensorView.create("output", [1], "int32")
    guard = TensorView.create("guard", [1], "int32")
    index = ir.Var("index", "int32")
    parameters = [output.tensor]
    prefix = []
    if alias_kind == "view":
        prefix.append(guard.declare(output.data))
    else:
        parameters.append(guard.tensor)
    branch = ir.If(guard.load(0) < 4, output.store(0, output.load(0) + 1), None)
    function = tirx.Function(parameters, [*prefix, ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, branch)])
    function = function.with_attr("global_symbol", "alias_guard").with_attr("tirx.noalias", alias_kind != "parameter")
    if alias_kind == "nonrestrict":
        function = function.with_attr("tl.non_restrict_param_indices", [0, 1])
    result = unswitch(function)
    assert not any(isinstance(statement, ir.If) for statement in result.body.seq)
    actual = runtime.tensor(np.array([0], dtype="int32"))
    executable = compile_tilelang(result, target="c")
    executable(actual) if alias_kind == "view" else executable(actual, actual)
    np.testing.assert_array_equal(actual.numpy(), [4])


@pytest.mark.parametrize("bound_predicate", [False, True])
@pytest.mark.parametrize("step", [1, -1])
def test_dynamic_zero_trip_loop_does_not_speculate_memory_guard(bound_predicate, step, *, compile_tilelang):
    output = TensorView.create("output", [8], "int32")
    guard = TensorView.create("guard", [1], "int32")
    extent = ir.Var("extent", "int32")
    index = ir.Var("index", "int32")
    condition = guard.load(0) > 0
    body = []
    if bound_predicate:
        predicate = ir.Var("predicate", "bool")
        body.append(ir.Bind(predicate, condition))
        condition = predicate
    body.append(ir.If(condition, output.store(index, 7), None))
    begin, end = (0, extent) if step > 0 else (extent - 1, -1)
    function = tirx.Function([output.tensor, guard.tensor, extent], ir.For(index, begin, end, step, ir.ForKind.DEFAULT, body))
    function = function.with_attr("global_symbol", "zero_trip_guard").with_attr("tirx.noalias", True)
    result = unswitch(function)
    outer = result.body.seq[0]
    assert isinstance(outer, ir.If)
    reads = []
    stmt_functor.post_order_visit(outer.condition, lambda node: reads.append(node) if isinstance(node, ir.TensorLoad) else None)
    assert not reads
    executable = compile_tilelang(result, target="c")
    for count in [0, 1, 8]:
        actual = runtime.tensor(np.zeros(8, dtype="int32"))
        executable(actual, runtime.tensor(np.array([1], dtype="int32")), count)
        expected = np.zeros(8, dtype="int32")
        expected[:count] = 7
        np.testing.assert_array_equal(actual.numpy(), expected)
