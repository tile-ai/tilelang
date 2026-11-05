"""No-op store analysis respects element types, lanes, and index side effects."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, target, tirx
from tvm_tirx.ir import prim

import tilelang
from tilelang.testing.ir import collect_nodes
from tilelang.testing.packed import compile_packed_function


DTYPES = ["int8", "uint32", "bool", "float32", "bfloat16", "float8_e4m3fn", "custom[tfloat32]32", "custom[float4_e2m1_unpacked]8"]
CPU = target.Target("c")


def _tensor(name, dtype):
    return ir.Var(name, tirx.TensorType((16,), dtype, align=0, layout="default"))


def _function(params, body):
    return tirx.Function(params, body).with_attr({"global_symbol": "main", "target": CPU})


def _remove(function):
    before = ir.save_json(function)
    result = tilelang.transform.RemoveNoOp()(ir.IRModule.from_expr(function))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(function) == before
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("source_kind", ["same_element", "other_tensor", "other_index"])
def test_store_identity_requires_same_location_with_matching_element_type(dtype, lanes, source_kind):
    output, other = _tensor("output", dtype), _tensor("other", dtype)
    index = ir.const(0, "int32") if lanes == 1 else prim.Ramp(0, 1, lanes)
    source = other if source_kind == "other_tensor" else output
    source_index = index + 4 if source_kind == "other_index" else index
    value = ir.TensorLoad(source, [source_index])
    function = _function([output, other], ir.TensorStore(output, [index], value))
    result = _remove(function)
    stores = collect_nodes(result, ir.TensorStore)
    assert len(stores) == (0 if source_kind == "same_element" else 1)
    if stores:
        assert stores[0].value.ty == value.ty
        ir.assert_structural_equal(stores[0], function.body.seq[0])


@pytest.mark.parametrize("dtype", ["int32", "bool", "custom[tfloat32]32"])
def test_effectful_index_expressions_cannot_prove_store_is_redundant(dtype):
    tensor = _tensor("tensor", dtype)
    # Reusing the Call IR object does not imply equal values on two evaluations.
    index = tirx.call_extern("next_storage_index", ty="int32")
    function = _function([tensor], ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index])))
    result = _remove(function)
    assert len(collect_nodes(result, ir.TensorStore)) == 1
    ir.assert_structural_equal(result, function)


@pytest.mark.parametrize("dtype", ["int32", "bool", "custom[tfloat32]32"])
def test_opaque_values_are_not_removed(dtype):
    tensor = _tensor("tensor", dtype)
    value = tirx.call_extern("next_storage_value", ty=dtype)
    function = _function([tensor], ir.TensorStore(tensor, [0], value))
    result = _remove(function)
    ir.assert_structural_equal(result, function)


@tilelang.testing.requires_llvm
def test_llvm_vector_copy_preserves_zero_signs_nan_payloads_and_infinities():
    source, output = _tensor("source", "float32"), _tensor("output", "float32")
    index = prim.Ramp(0, 1, 4)
    llvm = target.Target({"kind": "llvm", "mcpu": "generic"})
    function = _function([source, output], ir.TensorStore(output, [index], ir.TensorLoad(source, [index])))
    result = _remove(function.with_attr("target", llvm))
    executable = tirx.compile(result, target=llvm)
    bits = np.array([0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000] + [0] * 12, dtype="uint32")
    actual = runtime.tensor(np.zeros(16, dtype="float32"))
    executable(runtime.tensor(bits.view("float32")), actual)
    np.testing.assert_array_equal(actual.numpy().view("uint32"), bits)


def test_cpu_scalar_copy_preserves_zero_signs_nan_payloads_and_infinities():
    source, output = _tensor("source", "float32"), _tensor("output", "float32")
    index = ir.Var("index", "int32")
    function = _function(
        [source, output],
        ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, ir.TensorStore(output, [index], ir.TensorLoad(source, [index]))),
    )
    result = _remove(function)
    bits = np.array([0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000] + [0] * 12, dtype="uint32")
    actual = runtime.tensor(np.zeros(16, dtype="float32"))
    compile_packed_function(result, target="c")(runtime.tensor(bits.view("float32")), actual)
    np.testing.assert_array_equal(actual.numpy().view("uint32"), bits)


def test_cpu_effectful_indices_keep_both_calls_and_the_write():
    calls = []

    @tvm_ffi.register_global_func("tilelang.testing.noop_next_index", override=True)
    def next_index():
        value = len(calls)
        calls.append(value)
        return value

    tensor = _tensor("tensor", "int32")
    index = tirx.call_packed("tilelang.testing.noop_next_index", ty="int32")
    function = _function([tensor], ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index])))
    result = _remove(function)
    assert len(collect_nodes(result, ir.TensorStore)) == 1
    executable = compile_packed_function(result, target="c")
    initial = np.arange(16, dtype="int32") * 13 + 7
    actual = runtime.tensor(initial)
    executable(actual)
    assert calls == [0, 1]
    # Whichever operand evaluates first, one of the two different elements
    # must be copied to the other. Merely retaining call effects is insufficient.
    values = actual.numpy()
    assert values[0] == values[1]
    assert values[0] in initial[:2]
    np.testing.assert_array_equal(values[2:], initial[2:])


@pytest.mark.parametrize("dtype", ["int32", "bool", "custom[tfloat32]32"])
def test_volatile_self_store_and_read_evaluation_are_retained(dtype):
    tensor = _tensor("tensor", dtype)
    load = ir.TensorLoad(tensor, [0])
    function = _function([tensor], [ir.TensorStore(tensor, [0], load), ir.Evaluate(load)])
    function = function.with_attr("volatile_scope", True)
    result = _remove(function)
    assert len(collect_nodes(result, ir.TensorStore)) == 1
    assert len(collect_nodes(result, ir.Evaluate)) == 1
    ir.assert_structural_equal(result, function)


def test_cleanup_does_not_enable_simplify_options_or_inline_bindings():
    tensor = _tensor("tensor", "int32")
    x, value = ir.Var("x", "int32"), ir.Var("value", "int32")
    function = _function(
        [tensor, x],
        [
            ir.Bind(value, x + 1),
            ir.TensorStore(tensor, [0], 7),
            ir.TensorStore(tensor, [1], ir.TensorLoad(tensor, [0]) + value),
        ],
    )
    with ir.transform.PassContext(
        config={
            "tl.Simplify": {"enable_simplify_let_inline": True, "propagate_knowns_to_simplify_expressions": True},
            "tirx.RemoveNoOp": {"max_simplification_steps": 100, "ignore_profiler_call": False},
        }
    ):
        result = _remove(function)
    assert len(collect_nodes(result, ir.Bind)) == 1
    assert len(collect_nodes(result, ir.TensorLoad)) == 1
    ir.assert_structural_equal(result, function)


@pytest.mark.parametrize("begin,end", [(0, 0), (3, 3), (5, 2)])
def test_zero_trip_loop_is_removed_before_visiting_its_body(begin, end):
    tensor = _tensor("tensor", "int32")
    i = ir.Var("i", "int32")
    # A raw zero divisor is valid to construct, but it must never be analyzed
    # or evaluated within the unreachable loop body.
    poison = prim.FloorDiv(i, ir.const(0, "int32"))
    function = _function([tensor], ir.For(i, begin, end, 1, ir.ForKind.DEFAULT, ir.TensorStore(tensor, [0], poison)))
    result = _remove(function)
    assert not collect_nodes(result, ir.For)
    assert not collect_nodes(result, ir.TensorStore)


def test_effectful_step_in_zero_trip_loop_is_not_dropped_or_visited_as_a_body():
    tensor = _tensor("tensor", "int32")
    i = ir.Var("i", "int32")
    step = tirx.call_extern("step_with_effect", ty="int32")
    poison = prim.FloorDiv(i, ir.const(0, "int32"))
    function = _function([tensor], ir.For(i, 0, 0, step, ir.ForKind.DEFAULT, ir.TensorStore(tensor, [0], poison)))
    result = _remove(function)
    loop = collect_nodes(result, ir.For)[0]
    assert loop.step.same_as(step)
    assert not collect_nodes(loop.body, ir.TensorStore)


def test_same_effectful_bound_object_does_not_prove_loop_is_empty():
    calls = []

    @tvm_ffi.register_global_func("tilelang.testing.noop_bound", override=True)
    def next_bound():
        value = 0 if not calls else 2
        calls.append(value)
        return value

    output = _tensor("output", "int32")
    i = ir.Var("i", "int32")
    bound = tirx.call_packed("tilelang.testing.noop_bound", ty="int32")
    loop = ir.For(i, bound, bound, 1, ir.ForKind.DEFAULT, ir.TensorStore(output, [i], i + 7))
    function = _function([output], loop)
    result = _remove(function)
    assert len(collect_nodes(result, ir.For)) == 1
    actual = runtime.tensor(np.full(16, -1, dtype="int32"))
    compile_packed_function(result, target="c")(actual)
    assert calls[:2] == [0, 2]
    np.testing.assert_array_equal(actual.numpy()[:2], [7, 8])
    np.testing.assert_array_equal(actual.numpy()[2:], [-1] * 14)


def test_effectful_condition_of_empty_branches_is_evaluated_once():
    calls = []

    @tvm_ffi.register_global_func("tilelang.testing.noop_condition", override=True)
    def condition():
        calls.append(True)
        return 1

    predicate = tirx.call_packed("tilelang.testing.noop_condition", ty="int32")
    function = _function([], ir.If(predicate, ir.Evaluate(0), ir.Evaluate(0)))
    result = _remove(function)
    assert not collect_nodes(result, ir.If)
    assert len(collect_nodes(result, ir.Evaluate)) == 1
    compile_packed_function(result, target="c")()
    assert calls == [True]


def test_live_loop_and_function_metadata_are_preserved():
    source, output = _tensor("source", "int32"), _tensor("output", "int32")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        0,
        16,
        1,
        ir.ForKind.DEFAULT,
        ir.TensorStore(output, [i], ir.TensorLoad(source, [i])),
        annotations={"cleanup_test": "preserve"},
    )
    function = _function([source, output], loop).with_attr("cleanup_test", "preserve")
    before = ir.save_json(function)
    assert ir.save_json(_remove(function)) == before


@pytest.mark.parametrize("dtype", ["bool", "custom[tfloat32]32"])
def test_hoisting_uses_native_prefix_and_typed_cleanup(dtype):
    source, output = _tensor("source", dtype), _tensor("output", dtype)
    n, i = ir.Var("n", "int32"), ir.Var("i", "int32")
    loop = ir.For(
        i,
        0,
        16,
        1,
        ir.ForKind.DEFAULT,
        ir.If(n > 0, ir.TensorStore(output, [i], ir.TensorLoad(source, [i])), None),
    )
    function = _function([source, output, n], loop)
    before = ir.save_json(function)
    pipeline = tilelang.transform.HoistIf()
    assert tuple(stage.info.name for stage in pipeline.passes) == (
        "tirx.InsertHoistedIf",
        "tirx.StmtSimplify",
        "tl.RemoveNoOp",
    )
    result = pipeline(ir.IRModule.from_expr(function))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(function) == before
    assert isinstance(result.body.seq[0], ir.If)
    assert len(collect_nodes(result, ir.TensorStore)) == 1


@pytest.mark.parametrize("changed", ["not_sequential", "extra_stage", "wrong_tail"])
def test_hoist_bridge_rejects_changed_upstream_composition(monkeypatch, changed):
    pipeline = tirx.transform.HoistIf()
    stages = list(pipeline.passes)
    if changed == "extra_stage":
        stages.insert(0, tirx.transform.StmtSimplify())
    elif changed == "wrong_tail":
        stages[-1] = tirx.transform.StmtSimplify()
    substitute = stages[0] if changed == "not_sequential" else ir.transform.Sequential(stages)
    monkeypatch.setattr(tirx.transform, "HoistIf", lambda: substitute)
    with pytest.raises(RuntimeError, match="review the TileLang cleanup boundary"):
        tilelang.transform.HoistIf()
