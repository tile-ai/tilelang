"""Six simplification controls preserve typed, effectful execution contracts."""

import itertools

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

import tilelang
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def _nodes(node, kind):
    result = []
    tvm_ffi.structural_walk(node, (kind, result.append), order="pre")
    return result


def _apply(function, **options):
    module = ir.IRModule({"main": function.with_attr("global_symbol", "main")})
    original = ir.save_json(module)
    with ir.transform.PassContext(config={"tl.Simplify": options}):
        result = tilelang.transform.Simplify()(module)
    assert ir.save_json(module) == original
    assert tirx.analysis.check_well_formed(result)
    return result["main"]


def _compile(function):
    return compile_packed_function(function, target="c")


def _buffer(name, shape=(1,), dtype="int32"):
    return ir.Var(name, tirx.TensorType(shape, dtype))


def _host_array(values, dtype="int32"):
    return runtime.tensor(np.asarray(values, dtype=dtype))


@pytest.mark.parametrize("enabled", [False, True])
def test_let_inline_option_retains_or_removes_definition_and_uses(enabled):
    output = _buffer("output", (4,))
    i, value = ir.Var("i", "int32"), ir.Var("value", "int32")
    function = tirx.Function(
        [output], [ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [ir.Bind(value, i + 1), ir.TensorStore(output, [i], value)])]
    )
    result = _apply(function, enable_simplify_let_inline=enabled)
    assert bool(_nodes(result, ir.Bind)) is not enabled
    if not enabled:
        assert _nodes(result, ir.TensorStore)[0].value.same_as(value)
    actual = _host_array([0] * 4)
    _compile(result)(actual)
    np.testing.assert_array_equal(actual.numpy(), [1, 2, 3, 4])


@pytest.mark.parametrize("enabled", [False, True])
def test_known_tensor_values_control_expression_replacement(enabled):
    source, output = _buffer("source"), _buffer("output")
    function = tirx.Function(
        [source, output], [ir.TensorStore(source, [0], 7), ir.TensorStore(output, [0], ir.TensorLoad(source, [0]) + 1)]
    )
    result = _apply(function, propagate_knowns_to_simplify_expressions=enabled)
    assert bool(_nodes(result, ir.TensorLoad)) is not enabled
    actual = _host_array([0])
    _compile(result)(_host_array([0]), actual)
    np.testing.assert_array_equal(actual.numpy(), [8])


@pytest.mark.parametrize("enabled", [False, True])
def test_known_condition_option_does_not_replace_unrelated_expression(enabled):
    source, output = _buffer("source"), _buffer("output")
    function = tirx.Function(
        [source, output],
        [
            ir.TensorStore(source, [0], 7),
            ir.If(
                ir.TensorLoad(source, [0]) == 7,
                [ir.TensorStore(output, [0], ir.TensorLoad(source, [0]) + 1)],
                [ir.TensorStore(output, [0], 99)],
            ),
        ],
    )
    result = _apply(function, propagate_knowns_to_prove_conditional=enabled)
    assert bool(_nodes(result, ir.If)) is not enabled
    # The expression option remains false, even when the branch is proven.
    assert _nodes(result, ir.TensorLoad)
    actual = _host_array([0])
    _compile(result)(_host_array([0]), actual)
    np.testing.assert_array_equal(actual.numpy(), [8])


@pytest.mark.parametrize("dtype,value", [("bool", False), ("bool", True), ("int32", -3), ("int32", 2), ("uint32", 0)])
@pytest.mark.parametrize("inline", [False, True])
def test_literal_condition_selects_one_branch_and_records_bindings_once(dtype, value, inline):
    x, output = ir.Var("x", "int32"), _buffer("output", (2,))
    # Same spelling, different identities in the two lexical branches.
    left, right = ir.Var("value", "int32"), ir.Var("value", "int32")
    function = tirx.Function(
        [x, output],
        [
            ir.If(
                ir.const(value, dtype),
                [ir.Bind(left, x + 1), ir.TensorStore(output, [0], left)],
                [ir.Bind(right, x - 1), ir.TensorStore(output, [0], right)],
            ),
            ir.TensorStore(output, [1], x),
        ],
    )
    result = _apply(function, enable_simplify_let_inline=inline)
    assert not _nodes(result, ir.If)
    bindings = _nodes(result, ir.Bind)
    assert len(bindings) == (0 if inline else 1)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(_apply(result, enable_simplify_let_inline=inline), result)
    actual = _host_array([0, 0])
    _compile(result)(7, actual)
    np.testing.assert_array_equal(actual.numpy(), [8 if value else 6, 7])


def test_false_condition_without_else_removes_its_side_effects():
    output = _buffer("output")
    function = tirx.Function(
        [output],
        [ir.If(ir.const(False), [ir.TensorStore(output, [0], 99)], None)],
    )
    result = _apply(function)
    assert not _nodes(result, ir.If)
    assert not _nodes(result, ir.TensorStore)
    actual = _host_array([-123])
    _compile(result)(actual)
    np.testing.assert_array_equal(actual.numpy(), [-123])


def test_other_tensor_store_invalidates_potentially_aliasing_fact():
    a, alias, output = _buffer("a"), _buffer("alias"), _buffer("output")
    function = tirx.Function(
        [a, alias, output], [ir.TensorStore(a, [0], 1), ir.TensorStore(alias, [0], 2), ir.TensorStore(output, [0], ir.TensorLoad(a, [0]))]
    )
    result = _apply(function, propagate_knowns_to_simplify_expressions=True)
    assert _nodes(result, ir.TensorLoad)
    actual, shared = _host_array([0]), _host_array([0])
    _compile(result)(shared, shared, actual)
    np.testing.assert_array_equal(actual.numpy(), [2])


def test_memory_load_binding_keeps_its_original_evaluation_point():
    a, output, snapshot = _buffer("a"), _buffer("output"), ir.Var("snapshot", "int32")
    function = tirx.Function(
        [a, output], [ir.Bind(snapshot, ir.TensorLoad(a, [0])), ir.TensorStore(a, [0], 9), ir.TensorStore(output, [0], snapshot)]
    )
    result = _apply(function, enable_simplify_let_inline=True, propagate_knowns_to_simplify_expressions=True)
    assert _nodes(result, ir.Bind)
    actual = _host_array([0])
    _compile(result)(_host_array([2]), actual)
    np.testing.assert_array_equal(actual.numpy(), [2])


@pytest.mark.parametrize("guard", [False, True])
def test_conditional_store_invalidates_prebranch_knowledge(guard):
    a, output, flag = _buffer("a"), _buffer("output"), ir.Var("flag", "bool")
    function = tirx.Function(
        [a, output, flag],
        [ir.TensorStore(a, [0], 1), ir.If(flag, [ir.TensorStore(a, [0], 2)], None), ir.TensorStore(output, [0], ir.TensorLoad(a, [0]))],
    )
    result = _apply(function, propagate_knowns_to_simplify_expressions=True)
    assert _nodes(result, ir.TensorLoad)
    actual = _host_array([0])
    _compile(result)(_host_array([0]), actual, guard)
    np.testing.assert_array_equal(actual.numpy(), [2 if guard else 1])


def test_foreign_call_invalidates_memory_knowledge():
    a, output = _buffer("a"), _buffer("output")
    function = tirx.Function(
        [a, output],
        [
            ir.TensorStore(a, [0], 1),
            ir.Evaluate(tirx.call_extern("abs", ir.const(-1), ty=ir.PrimType("int32"))),
            ir.TensorStore(output, [0], ir.TensorLoad(a, [0])),
        ],
    )
    result = _apply(function, propagate_knowns_to_simplify_expressions=True)
    assert _nodes(result, ir.TensorLoad)
    actual = _host_array([0])
    _compile(result)(_host_array([0]), actual)
    np.testing.assert_array_equal(actual.numpy(), [1])


def test_volatile_scope_does_not_propagate_stored_value():
    @T.prim_func
    def kernel(a: T.Tensor((1,), "int32"), output: T.Tensor((1,), "int32")):
        with T.attr(a, "volatile_scope", 1):
            a[0] = 1
            output[0] = a[0]

    result = _apply(kernel, propagate_knowns_to_simplify_expressions=True)
    assert _nodes(result, ir.TensorLoad)


@pytest.mark.parametrize("enabled", [False, True])
def test_conjunctive_form_preserves_complete_boolean_truth_table(enabled):
    a, b, c = [ir.Var(name, "bool") for name in ("a", "b", "c")]
    output = _buffer("output")
    expression = ir.prim.Or(ir.prim.And(a, b), c)
    function = tirx.Function([a, b, c, output], [ir.TensorStore(output, [0], ir.prim.Cast("int32", expression))])
    result = _apply(function, convert_boolean_to_and_of_ors=enabled)
    if enabled:
        for disjunction in _nodes(result, ir.prim.Or):
            assert not _nodes(disjunction, ir.prim.And)
    executable = _compile(result)
    for av, bv, cv in itertools.product([False, True], repeat=3):
        actual = _host_array([0])
        executable(av, bv, cv, actual)
        np.testing.assert_array_equal(actual.numpy(), [int((av and bv) or cv)])


@pytest.mark.parametrize("enabled", [False, True])
def test_boolean_operand_constraint_rewrites_bounded_division(enabled):
    x, output = ir.Var("x", "uint32"), _buffer("output")
    expression = ir.prim.And(x < ir.const(3, "uint32"), ir.prim.FloorDiv(x, ir.const(3, "uint32")) == ir.const(0, "uint32"))
    function = tirx.Function([x, output], [ir.TensorStore(output, [0], ir.prim.Cast("int32", expression))])
    result = _apply(function, apply_constraints_to_boolean_branches=enabled)
    if enabled:
        assert not _nodes(result, ir.prim.FloorDiv)
    executable = _compile(result)
    for value in [0, 1, 2, 3, 4, 255, 256, (1 << 32) - 1]:
        actual = _host_array([0])
        executable(value, actual)
        np.testing.assert_array_equal(actual.numpy(), [int(value < 3)])


@pytest.mark.parametrize("enabled", [False, True])
def test_transitive_proofs_use_scoped_constraints_without_leaking(enabled):
    x, y, z = [ir.Var(name, "int32") for name in ("x", "y", "z")]
    output = _buffer("output", (2,))
    function = tirx.Function(
        [x, y, z, output],
        [
            ir.If(x < y, [ir.If(y < z, [ir.TensorStore(output, [0], ir.prim.Cast("int32", x < z))], None)], None),
            ir.TensorStore(output, [1], ir.prim.Cast("int32", x < z)),
        ],
    )
    result = _apply(function, transitively_prove_inequalities=enabled)
    if enabled:
        assert any(isinstance(store.value, ir.IntImm) and int(store.value) == 1 for store in _nodes(result, ir.TensorStore))
    executable = _compile(result)
    for xv, yv, zv in [(1, 2, 3), (3, 2, 1), (1, 3, 2), (-3, -2, -1), (0, 0, 0)]:
        actual = _host_array([0, 0])
        executable(xv, yv, zv, actual)
        np.testing.assert_array_equal(actual.numpy(), [int(xv < yv < zv), int(xv < zv)])


@pytest.mark.xfail(strict=True, reason="Inherited upstream narrowing-cast proof defect; deferred from this rebase")
def test_optional_proofs_do_not_erase_narrow_cast_counterexample():
    x, output = ir.Var("x", "uint32"), _buffer("output")
    predicate = ir.prim.Cast("uint32", ir.prim.Cast("uint8", x)) == x
    function = tirx.Function([x, output], [ir.TensorStore(output, [0], ir.prim.Cast("int32", predicate))])
    result = _apply(
        function, transitively_prove_inequalities=True, convert_boolean_to_and_of_ors=True, apply_constraints_to_boolean_branches=True
    )
    executable = _compile(result)
    for value in [0, 1, 255, 256, 257, 65535]:
        actual = _host_array([0])
        executable(value, actual)
        np.testing.assert_array_equal(actual.numpy(), [int((value & 255) == value)])


def test_invalid_configuration_key_is_reported():
    with (
        pytest.raises((AttributeError, TypeError, ValueError), match="unknown_setting"),
        ir.transform.PassContext(config={"tl.Simplify": {"unknown_setting": True}}),
    ):
        pass


def test_while_condition_must_be_reloaded_on_every_iteration():
    a, output = _buffer("a"), _buffer("output")
    function = tirx.Function(
        [a, output],
        [
            ir.TensorStore(a, [0], 0),
            ir.While(
                ir.TensorLoad(a, [0]) < 3,
                [
                    ir.TensorStore(a, [0], ir.TensorLoad(a, [0]) + 1),
                ],
            ),
            ir.TensorStore(output, [0], ir.TensorLoad(a, [0])),
        ],
    )
    result = _apply(function, propagate_knowns_to_simplify_expressions=True, propagate_knowns_to_prove_conditional=True)
    loop = _nodes(result, ir.While)[0]
    # Check before execution: replacing this load by the entry fact loops forever.
    assert _nodes(loop.condition, ir.TensorLoad)
    actual = _host_array([0])
    _compile(result)(_host_array([17]), actual)
    np.testing.assert_array_equal(actual.numpy(), [3])


@pytest.mark.parametrize("inline", [False, True])
def test_dead_binding_chain_removes_unobserved_nonvolatile_load(inline):
    a, output = _buffer("a"), _buffer("output")
    value, unused = ir.Var("value", "int32"), ir.Var("unused", "int32")
    function = tirx.Function(
        [a, output],
        [
            ir.Bind(value, ir.TensorLoad(a, [0])),
            ir.Bind(unused, value + 1),
            ir.TensorStore(output, [0], 7),
        ],
    )
    result = _apply(function, enable_simplify_let_inline=inline)
    assert not _nodes(result, ir.Bind)
    assert not _nodes(result, ir.TensorLoad)
    actual = _host_array([0])
    _compile(result)(_host_array([123]), actual)
    np.testing.assert_array_equal(actual.numpy(), [7])


def test_unused_consumer_does_not_delete_effectful_producer():
    output = _buffer("output")
    value, unused = ir.Var("value", "int32"), ir.Var("unused", "int32")
    operation = tirx.call_extern("abs", ir.const(-1), ty=ir.PrimType("int32"))
    function = tirx.Function([output], [ir.Bind(value, operation), ir.Bind(unused, value + 1), ir.TensorStore(output, [0], 7)])
    result = _apply(function)
    bindings = _nodes(result, ir.Bind)
    assert len(bindings) == 1
    assert bindings[0].value.same_as(operation)
    actual = _host_array([0])
    _compile(result)(actual)
    np.testing.assert_array_equal(actual.numpy(), [7])


def test_scalar_parameters_survive_explicit_tensor_argument_cleanup():
    dead, output = _buffer("dead"), _buffer("output")
    scalar = ir.Var("scalar", "int32")
    function = tirx.Function([dead, output, scalar], [ir.TensorStore(output, [0], 7)])
    result = tilelang.transform.Simplify(simplify_arguments=True)(ir.IRModule({"main": function.with_attr("global_symbol", "main")}))[
        "main"
    ]
    assert [parameter.name for parameter in result.params] == ["output", "scalar"]
    assert len(result.ty.arg_types) == len(result.params)
    assert tvm_ffi.structural_equal(result.ty.arg_types[0], output.ty)
    actual = _host_array([0])
    _compile(result)(actual, 123)
    np.testing.assert_array_equal(actual.numpy(), [7])


@pytest.mark.parametrize("constant", [False, True])
def test_binding_used_only_in_tensor_shape_is_preserved_or_retyped(constant):
    a, output, size = _buffer("a"), _buffer("output"), ir.Var("size", "int32")
    scratch_type = tirx.TensorType((size,), "int32", scope="local")
    scratch = ir.Var("scratch", scratch_type)
    function = tirx.Function(
        [a, output],
        [
            ir.Bind(size, ir.const(4) if constant else ir.TensorLoad(a, [0])),
            ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch_type])),
            ir.TensorStore(scratch, [0], 9),
            ir.TensorStore(output, [0], ir.TensorLoad(scratch, [0])),
        ],
    )
    result = _apply(function)
    bindings = _nodes(result, ir.Bind)
    if not constant:
        assert any(binding.var.same_as(size) for binding in bindings)
    else:
        allocation = next(binding for binding in bindings if isinstance(binding.var.ty, tirx.TensorType))
        # Tensor contracts are preserved; a metadata-only definition remains
        # live even after all ordinary scalar uses were inlined.
        assert allocation.var.ty.shape[0].same_as(size)
        assert any(binding.var.same_as(size) for binding in bindings)
        assert tvm_ffi.structural_equal(allocation.var.ty, allocation.value.ty)
    actual = _host_array([0])
    _compile(result)(_host_array([4]), actual)
    np.testing.assert_array_equal(actual.numpy(), [9])


@pytest.mark.parametrize("volatile", [False, True])
def test_same_address_load_store_only_removed_for_nonvolatile_memory(volatile):
    @T.prim_func
    def kernel(a: T.Tensor((1,), "int32")):
        if volatile:
            with T.attr(a, "volatile_scope", 1):
                a[0] = a[0]
        else:
            a[0] = a[0]

    result = _apply(kernel)
    assert bool(_nodes(result, ir.TensorStore)) is volatile
    assert bool(_nodes(result, ir.TensorLoad)) is volatile


def test_original_macro_ignored_argument_no_longer_reads_uninitialized_local():
    # Kernel spelling from the frozen test_simplify_removes_unused_macro_bind_chain.
    @T.macro
    def ignore_rebound_argument(value):
        value = value + 1.0
        value = value + 2.0
        return 7.0

    @T.prim_func
    def before(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_var("float32")
            B[0] = ignore_rebound_argument(x)

    result = _apply(before)
    assert not _nodes(result, ir.TensorLoad)


@pytest.mark.parametrize("site", ["function", "call"])
@pytest.mark.parametrize("constant", [False, True])
def test_binding_referenced_only_by_metadata_remains_live(site, constant):
    source, output = _buffer("source"), _buffer("output")
    tag = ir.Var("tag", "int32")
    operation = tirx.call_extern("abs", ir.const(-7), ty=ir.PrimType("int32"))
    metadata = {"test.metadata": {"nested": [tag]}}
    if site == "call":
        operation = ir.Call(operation.op, operation.args, attrs=metadata, ty=operation.ty)
    function = tirx.Function(
        [source, output],
        [
            ir.Bind(tag, ir.const(4) if constant else ir.TensorLoad(source, [0])),
            ir.TensorStore(output, [0], operation),
        ],
    )
    if site == "function":
        function = function.with_attr("test.metadata", metadata["test.metadata"])
    result = _apply(function)
    assert any(binding.var.same_as(tag) for binding in _nodes(result, ir.Bind))
    actual = _host_array([0])
    _compile(result)(_host_array([4]), actual)
    np.testing.assert_array_equal(actual.numpy(), [7])


def test_unused_volatile_read_remains_after_dead_consumer_cleanup():
    @T.prim_func
    def kernel(a: T.Tensor((1,), "int32"), output: T.Tensor((1,), "int32")):
        with T.attr(a, "volatile_scope", 1):
            snapshot = a[0]
            unused = snapshot + 1  # noqa: F841 - deliberately dead IR binding
            output[0] = 7

    result = _apply(kernel)
    bindings = _nodes(result, ir.Bind)
    assert len(bindings) == 1
    assert isinstance(bindings[0].value, ir.TensorLoad)


@pytest.mark.parametrize("enabled", [False, True])
def test_nonvolatile_boolean_loads_can_use_conjunctive_form(enabled):
    inputs, output = _buffer("inputs", (3,), "bool"), _buffer("output")
    expression = ir.prim.Or(ir.TensorLoad(inputs, [0]), ir.prim.And(ir.TensorLoad(inputs, [1]), ir.TensorLoad(inputs, [2])))
    function = tirx.Function([inputs, output], [ir.TensorStore(output, [0], ir.prim.Cast("int32", expression))])
    result = _apply(function, convert_boolean_to_and_of_ors=enabled)
    if enabled:
        assert all(not _nodes(disjunction, ir.prim.And) for disjunction in _nodes(result, ir.prim.Or))
    else:
        assert any(_nodes(disjunction, ir.prim.And) for disjunction in _nodes(result, ir.prim.Or))
    executable = _compile(result)
    for av, bv, cv in itertools.product([False, True], repeat=3):
        actual = _host_array([0])
        executable(_host_array([av, bv, cv], "bool"), actual)
        np.testing.assert_array_equal(actual.numpy(), [int(av or (bv and cv))])
