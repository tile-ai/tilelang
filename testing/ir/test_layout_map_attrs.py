"""Layout declarations follow the owning pass's variable and scope context."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir.transform import PassContext

import tilelang
from tilelang.layout import Fragment, Layout, PartialFragment


KINDS = ["layout", "fragment", "partial"]
if tvm_ffi.get_global_func("tl.make_ascend_sf_layout", allow_missing=True) is not None:
    KINDS.append("ascend")


def integer(value):
    return ir.IntImm("int32", value)


def note(layouts):
    return tvm_ffi.get_global_func("tl.MakeLayoutMapAnnotation")(layouts, False, ir.UnknownLoc())


def renew(function):
    return tvm_ffi.get_global_func("tl.RenewDef")(function)


def substitute(statement, replacements):
    return tvm_ffi.get_global_func("tl.SubstituteLayoutMap")(statement, replacements)


def check_scopes(function):
    # A throwing checker may return None; a boolean checker must report success.
    assert tvm_ffi.get_global_func("tl.CheckLayoutMapScopes")(function) is not False


def nodes(root, cls):
    found = []
    tvm_ffi.structural_walk(root, [(cls, found.append)])
    return found


def layout_calls(root):
    op = ir.Op.get("tl.layout_map")
    return [call for call in nodes(root, ir.Call) if call.op.same_as(op)]


def entry(root):
    calls = layout_calls(root)
    assert len(calls) == 1
    mapping = calls[0].attrs["layout_map"]
    assert len(mapping) == 1
    return next(iter(mapping.items()))


def contains_identity(values, value):
    return any(candidate.same_as(value) for candidate in values)


def tensor(name="A", shape=(16,), scope="local"):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty=value.ty, ty_args=[value.ty]))


def make_layout(kind, n):
    shape = (n, 16, 32)
    if kind == "layout":
        return Layout(shape, lambda i, j, k: (i, j, (k + n) % 32))
    if kind == "ascend":
        from tilelang.layout import make_ascend_sf_layout

        return make_ascend_sf_layout(ir.Var("source", tirx.TensorType(shape, "uint8", scope="shared.l1")))
    factory = PartialFragment if kind == "partial" else Fragment
    return factory(
        shape,
        forward_index_fn=lambda i, j, k: (i, j),
        forward_thread_fn=lambda i, j, k, rep: k + rep * 32,
        replicate=2,
    )


def function_fixture(kind="layout", forward_shape=False):
    # An external shape and the internal coordinate deliberately share a name.
    n = ir.Var("i0", "int32")
    a = ir.Var("A", tirx.TensorType((n, 16, 32), "uint8", scope="shared.l1"))
    layout = make_layout(kind, n)
    params = [a, n] if forward_shape else [n, a]
    function = tirx.Function(
        params,
        [note({a: layout}), ir.TensorStore(a, [integer(0)] * 3, ir.IntImm("uint8", 1))],
        ir.PrimType("void"),
    )
    return function, n, a, layout


def numerical_map(layout, capture, value=8):
    coordinates = [0, 1, 2, 3] if isinstance(layout, Fragment) else [1, 2, 3]
    expressions = layout.map_forward_index(coordinates)
    if isinstance(layout, Fragment):
        expressions = [*expressions, *layout.map_forward_thread([1, 1, 2, 3])]
    analyzer = sym.Analyzer()
    result = []
    for expression in expressions:
        replaced = tvm_ffi.structural_map(
            expression,
            [(ir.Var, lambda var: integer(value) if var.same_as(capture) else var)],
        )
        result.append(int(analyzer.simplify(replaced)))
    return result


@pytest.mark.parametrize("kind", KINDS)
def test_renew_updates_key_capture_and_coordinate_definitions_together(kind):
    before, n, a, original_layout = function_fixture(kind)
    snapshot = ir.save_json(before)
    left, right = renew(before), renew(before)
    for after in (left, right):
        new_n, new_a = after.params
        key, layout = entry(after)
        assert not new_n.same_as(n)
        assert not new_a.same_as(a)
        assert key.same_as(new_a)
        assert key.ty.shape[0].same_as(new_n)
        assert layout.get_input_shape()[0].same_as(new_n)
        assert nodes(after, ir.TensorStore)[0].dest.same_as(new_a)
        assert type(layout) is type(original_layout)
        assert numerical_map(layout, new_n) == numerical_map(original_layout, n)
        assert len(layout.coordinate_vars) == len(original_layout.coordinate_vars)
        for old, new in zip(original_layout.coordinate_vars, layout.coordinate_vars):
            assert not old.same_as(new)
        raw_uses = nodes(layout.forward_index, ir.Var)
        if isinstance(layout, Fragment):
            raw_uses += nodes(layout.forward_thread, ir.Var)
            assert int(layout.replicate_size) == int(original_layout.replicate_size)
        if isinstance(layout, PartialFragment):
            assert int(layout.combine_size) == int(original_layout.combine_size)
        if kind == "ascend":
            assert layout.kind == original_layout.kind
        assert all(var.same_as(new_n) or contains_identity(layout.coordinate_vars, var) for var in raw_uses)
        check_scopes(after)
        ir.assert_structural_equal(before, after)
        assert tvm_ffi.structural_hash(before) == tvm_ffi.structural_hash(after)
    assert not left.params[1].same_as(right.params[1])
    for left_var, right_var in zip(entry(left)[1].coordinate_vars, entry(right)[1].coordinate_vars):
        assert not left_var.same_as(right_var)
    assert ir.save_json(before) == snapshot


def test_renew_accepts_forward_shape_parameter_order():
    before, n, a, _ = function_fixture(forward_shape=True)
    check_scopes(before)
    after = renew(before)
    new_a, new_n = after.params
    key, layout = entry(after)
    assert key.same_as(new_a)
    assert new_a.ty.shape[0].same_as(new_n)
    assert layout.get_input_shape()[0].same_as(new_n)
    assert not new_a.same_as(a) and not new_n.same_as(n)
    check_scopes(after)
    ir.assert_structural_equal(before, after)


@pytest.mark.parametrize("kind", KINDS)
def test_substitute_rebuilds_call_key_and_capture_in_one_context(kind):
    before, n, a, original_layout = function_fixture(kind)
    snapshot = ir.save_json(before)
    new_n = ir.Var("i0", "int32")
    new_a = ir.Var("A", tirx.TensorType((new_n, 16, 32), "uint8", scope="shared.l1"))
    body = substitute(before.body, {n: new_n, a: new_a})
    after = tirx.Function([new_n, new_a], body, ir.PrimType("void"))
    key, layout = entry(after)
    assert key.same_as(new_a)
    assert layout.get_input_shape()[0].same_as(new_n)
    assert nodes(after, ir.TensorStore)[0].dest.same_as(new_a)
    assert type(layout) is type(original_layout)
    assert numerical_map(layout, new_n) == numerical_map(original_layout, n)
    # Ordinary substitution preserves unrelated local coordinate definitions.
    assert all(old.same_as(new) for old, new in zip(original_layout.coordinate_vars, layout.coordinate_vars))
    check_scopes(after)
    assert ir.save_json(before) == snapshot


def test_substitute_shape_rebuilds_local_definition_and_all_uses():
    n = ir.Var("n", "int32")
    a = tensor(shape=(n,))
    layout = Layout((n,), lambda i: i + n)
    before = ir.SeqStmt([allocate(a), note({a: layout}), ir.TensorStore(a, [integer(0)], integer(1))])
    snapshot = ir.save_json(before)
    after = substitute(before, {n: integer(8)})
    binding = nodes(after, ir.Bind)[0]
    key, updated_layout = entry(after)
    assert not binding.var.same_as(a)
    assert key.same_as(binding.var)
    assert nodes(after, ir.TensorStore)[0].dest.same_as(binding.var)
    assert int(binding.var.ty.shape[0]) == 8
    assert int(binding.value.ty.shape[0]) == 8
    assert int(updated_layout.get_input_shape()[0]) == 8
    assert int(sym.Analyzer().simplify(updated_layout.map_forward_index([3])[0])) == 11
    check_scopes(tirx.Function([], after, ir.PrimType("void")))
    assert ir.save_json(before) == snapshot


def test_substitute_uses_identity_for_same_named_keys_and_captures():
    first_n, second_n, new_n = (ir.Var("n", "int32") for _ in range(3))
    a, b = tensor("A", (first_n,)), tensor("A", (second_n,))
    new_a = tensor("A", (new_n,))
    first = Layout((first_n,), lambda i: i + first_n)
    second = Layout((second_n,), lambda i: i + second_n)
    before = note({a: first, b: second})
    after = substitute(before, {first_n: new_n, a: new_a})
    mapping = layout_calls(after)[0].attrs["layout_map"]
    assert len(mapping) == 2
    assert mapping[new_a].get_input_shape()[0].same_as(new_n)
    assert mapping[b].get_input_shape()[0].same_as(second_n)
    assert not contains_identity(mapping.keys(), a)
    assert mapping[b].same_as(second)
    check_scopes(tirx.Function([new_n, second_n, new_a, b], [after], ir.PrimType("void")))


@pytest.mark.parametrize("replacement", ["empty", "irrelevant"])
def test_noop_substitution_preserves_original_objects(replacement):
    before, _, _, layout = function_fixture()
    replacements = {} if replacement == "empty" else {ir.Var("absent", "int32"): integer(7)}
    after = substitute(before.body, replacements)
    assert after.same_as(before.body)
    assert layout_calls(after)[0].same_as(layout_calls(before)[0])
    assert entry(after)[1].same_as(layout)


@pytest.mark.parametrize("kind", ["same_object", "equal_layout", "conflicting_layout"])
def test_low_level_substitute_rejects_every_unadjudicated_key_collision(kind):
    a, b = tensor("A"), tensor("B")
    first = Layout((16,), lambda i: i)
    if kind == "same_object":
        second = first
    elif kind == "equal_layout":
        second = Layout((16,), lambda i: i)
    else:
        second = Layout((16,), lambda i: 15 - i)
    before = note({a: first, b: second})
    snapshot = ir.save_json(before)
    with pytest.raises(ValueError, match="layout_map"):
        substitute(before, {b: a})
    assert ir.save_json(before) == snapshot


@pytest.mark.parametrize("replacement", ["scalar_var", "constant"])
def test_substitute_rejects_a_key_that_stops_being_a_tensor(replacement):
    before, _, a, _ = function_fixture()
    value = ir.Var("A", "int32") if replacement == "scalar_var" else integer(0)
    with pytest.raises((TypeError, ValueError)):
        substitute(before.body, {a: value})


def test_renew_keeps_free_captures_and_separates_local_copies():
    external = ir.Var("i0", "int32")
    a = tensor(shape=(external,))
    before = tirx.Function(
        [],
        [allocate(a), note({a: Layout((external,), lambda i: i + external)})],
        ir.PrimType("void"),
    )
    left, right = renew(before), renew(before)
    for after in (left, right):
        key, layout = entry(after)
        assert key.same_as(nodes(after, ir.Bind)[0].var)
        assert key.ty.shape[0].same_as(external)
        assert layout.get_input_shape()[0].same_as(external)
    assert not entry(left)[0].same_as(entry(right)[0])
    # This temporary function deliberately has a free capture, as used while
    # duplicating a loop body. The whole-function scope checker is not applied.


def test_shared_layout_can_be_renewed_without_mutating_other_owners():
    n = ir.Var("n", "int32")
    a, b = tensor("A", (n,)), tensor("B", (n,))
    shared = Layout((n,), lambda i: i + n)
    before = tirx.Function([n, a, b], [note({a: shared}), note({b: shared})], ir.PrimType("void"))
    snapshot = ir.save_json(before)
    after = renew(before)
    calls = layout_calls(after)
    assert len(calls) == 2
    for index, call in enumerate(calls, start=1):
        mapping = call.attrs["layout_map"]
        key, layout = next(iter(mapping.items()))
        assert key.same_as(after.params[index])
        assert layout.get_input_shape()[0].same_as(after.params[0])
        expression = layout.map_forward_index([3])[0]
        concrete = tvm_ffi.structural_map(
            expression,
            [(ir.Var, lambda var: integer(8) if var.same_as(after.params[0]) else var)],
        )
        assert int(sym.Analyzer().simplify(concrete)) == 11
    check_scopes(after)
    assert ir.save_json(before) == snapshot
    for call in layout_calls(before):
        assert next(iter(call.attrs["layout_map"].values())).same_as(shared)


def test_simplify_keeps_attrs_only_bind_and_tensor_uses():
    n, captured, dead = (ir.Var(name, "int32") for name in ("n", "captured", "dead"))
    a, unused = tensor("A"), tensor("unused")
    before = tirx.Function(
        [n, a, unused],
        [
            ir.Bind(captured, n + 1),
            ir.Bind(dead, n + 2),
            note({a: Layout((16,), lambda i: i + captured)}),
        ],
        ir.PrimType("void"),
    )
    with PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        after = tilelang.transform.Simplify(simplify_arguments=True)(ir.IRModule({"main": before}))["main"]
    bindings = nodes(after, ir.Bind)
    assert any(binding.var.same_as(captured) for binding in bindings)
    assert not any(binding.var.same_as(dead) for binding in bindings)
    assert contains_identity(after.params, a)
    assert not contains_identity(after.params, unused)
    key, layout = entry(after)
    assert key.same_as(a)
    assert contains_identity(nodes(layout.forward_index, ir.Var), captured)
    check_scopes(after)


def test_actual_unroll_renews_layout_keys_and_substitutes_iteration():
    i = ir.Var("i", "int32")
    a = tensor()
    loop = ir.For(
        i,
        integer(0),
        integer(2),
        integer(1),
        ir.ForKind.UNROLLED,
        [
            allocate(a),
            note({a: Layout((16,), lambda j: j + i)}),
            ir.TensorStore(a, [integer(0)], i),
        ],
        annotations={"pragma_unroll_explicit": True},
    )
    before = tirx.Function([], [loop], ir.PrimType("void"))
    snapshot = ir.save_json(before)
    after = tilelang.transform.UnrollLoop()(ir.IRModule({"main": before}))["main"]
    calls, bindings, stores = layout_calls(after), nodes(after, ir.Bind), nodes(after, ir.TensorStore)
    assert len(calls) == len(bindings) == len(stores) == 2
    assert not nodes(after, ir.For)
    keys = [next(iter(call.attrs["layout_map"])) for call in calls]
    assert not keys[0].same_as(keys[1])
    for iteration, (call, binding, store) in enumerate(zip(calls, bindings, stores)):
        key = next(iter(call.attrs["layout_map"]))
        layout = call.attrs["layout_map"][key]
        assert key.same_as(binding.var) and store.dest.same_as(binding.var)
        assert int(sym.Analyzer().simplify(layout.map_forward_index([3])[0])) == 3 + iteration
        assert int(store.value) == iteration
        assert not contains_identity(nodes(layout.forward_index, ir.Var), i)
    check_scopes(after)
    assert ir.save_json(before) == snapshot


def test_scope_checker_accepts_local_definitions_before_layout_uses():
    a = tensor()
    before = tirx.Function([], [allocate(a), note({a: Layout((16,), lambda i: i)})], ir.PrimType("void"))
    check_scopes(before)
    check_scopes(renew(before))


@pytest.mark.parametrize("problem", ["missing_key", "same_named_key", "missing_capture", "before_definition", "escaped_local"])
def test_scope_checker_rejects_invalid_attrs_only_references(problem):
    a = tensor()
    n, loop_var = ir.Var("n", "int32"), ir.Var("i", "int32")
    ordinary = Layout((16,), lambda i: i)
    if problem == "missing_key":
        before = tirx.Function([], [note({a: ordinary})], ir.PrimType("void"))
    elif problem == "same_named_key":
        impostor = tensor("A")
        before = tirx.Function([a], [note({impostor: ordinary})], ir.PrimType("void"))
    elif problem == "missing_capture":
        before = tirx.Function([a], [note({a: Layout((16,), lambda i: i + n)})], ir.PrimType("void"))
    elif problem == "before_definition":
        before = tirx.Function([], [note({a: ordinary}), allocate(a)], ir.PrimType("void"))
    else:
        loop = ir.For(loop_var, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [allocate(a)])
        before = tirx.Function([], [loop, note({a: ordinary})], ir.PrimType("void"))
    with pytest.raises(ValueError, match="layout_map"):
        check_scopes(before)


def test_native_renew_remains_a_negative_control():
    before, n, a, _ = function_fixture()
    after = tirx.renew_def(before)
    key, layout = entry(after)
    assert key.same_as(a)
    assert not key.same_as(after.params[1])
    assert layout.get_input_shape()[0].same_as(n)
    assert not after.params[0].same_as(n)
    with pytest.raises(ValueError, match="layout_map"):
        check_scopes(after)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("forward_shape", [False, True], ids=["shape_first", "tensor_first"])
def test_function_script_preserves_dynamic_layout_identity(kind, forward_shape):
    from tvm_tirx.script import from_source
    from tvm_tirx.script import tirx as T

    before, n, a, layout = function_fixture(kind, forward_shape=forward_shape)
    snapshot = ir.save_json(before)
    # Parsing receives namespaces only; original Vars and Layouts are not
    # supplied as globals. "i0" also collides with a local coordinate spelling.
    after = from_source(before.script(), extra_vars={"T": T, "tilelang": tilelang})
    n_index, a_index = (1, 0) if forward_shape else (0, 1)
    new_n, new_a = after.params[n_index], after.params[a_index]
    key, restored_layout = entry(after)
    assert not new_n.same_as(n) and not new_a.same_as(a)
    assert key.same_as(new_a)
    assert key.ty.shape[0].same_as(new_n)
    assert restored_layout.get_input_shape()[0].same_as(new_n)
    assert nodes(after, ir.TensorStore)[0].dest.same_as(new_a)
    assert type(restored_layout) is type(layout)
    if isinstance(layout, Fragment):
        assert int(restored_layout.replicate_size) == int(layout.replicate_size)
    if isinstance(layout, PartialFragment):
        assert int(restored_layout.combine_size) == int(layout.combine_size)
    if kind == "ascend":
        assert restored_layout.kind == layout.kind
    for value in (8, 19):
        assert numerical_map(restored_layout, new_n, value) == numerical_map(layout, n, value)
    raw_uses = nodes(restored_layout.forward_index, ir.Var)
    if isinstance(restored_layout, Fragment):
        raw_uses += nodes(restored_layout.forward_thread, ir.Var)
    assert all(var.same_as(new_n) or contains_identity(restored_layout.coordinate_vars, var) for var in raw_uses)
    check_scopes(after)
    ir.assert_structural_equal(before, after)
    assert tvm_ffi.structural_hash(before) == tvm_ffi.structural_hash(after)
    assert ir.save_json(before) == snapshot
