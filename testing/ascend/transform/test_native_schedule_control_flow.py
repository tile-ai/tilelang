"""Native Ascend schedule normalization/restoration retains loop semantics."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function
from testing.ascend._block import attribute as _scope_attribute


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape=(1,), dtype="int32", scope="global"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope))


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def attribute(key, value, body):
    return _scope_attribute(None, key, value, body)


def root(body):
    return attribute("tl.sblock", {"name": "tilelang_root", "no_realize": True}, body)


def apply(name, function):
    assert tirx.analysis.check_well_formed(function)
    saved = ir.save_json(function)
    result = tvm_ffi.get_global_func("tl.transform." + name)()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == saved
    assert tirx.analysis.check_well_formed(result)
    for call in objects(result, ir.Call):
        call.validate()
    return result


def compile_native(function, tmp_path):
    function = function.with_attr("global_symbol", "main").with_attr("target", tvm_tirx.target.Target("c"))
    module = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(ir.IRModule({"main": function}))
    (tmp_path / "lowered.json").write_text(ir.save_json(module))
    return compile_packed_function(module)


def test_while_condition_is_repeated_and_break_is_native(tmp_path):
    count, output = tensor("count"), tensor("output", (16,))
    index = ir.Var("index", "int32")
    condition = ir.TensorLoad(count, [0]) < 7
    body = [ir.Bind(index, ir.TensorLoad(count, [0])), ir.TensorStore(output, [index], index + 2), ir.TensorStore(count, [0], index + 1)]
    before = tirx.Function([count, output], [root([ir.While(condition, body)])])
    normalized = apply("NormalizeControlFlowForSchedule", before)
    assert not objects(normalized, ir.While)
    assert len(objects(normalized, ir.Break)) == 1
    synthetic = objects(normalized, ir.For)[0]
    assert int(synthetic.end.value) == 2
    assert isinstance(synthetic.body.seq[0], ir.Bind)
    assert objects(synthetic.body.seq[0].value, ir.TensorLoad)
    restored = apply("RestoreWhileLoops", normalized)
    executable = compile_native(restored, tmp_path)
    state = tvm_tirx.runtime.tensor(np.array([0], dtype="int32"))
    result = tvm_tirx.runtime.tensor(np.full(16, -1, dtype="int32"))
    executable(state, result)
    np.testing.assert_array_equal(state.numpy(), np.array([7], dtype="int32"))
    expected = np.full(16, -1, dtype="int32")
    expected[:7] = np.arange(7, dtype="int32") + 2
    np.testing.assert_array_equal(result.numpy(), expected)


def test_residual_iteration_counter_advances_on_continue(tmp_path):
    output = tensor("output", (12,))
    w = ir.Var("w", "int32")
    body = [ir.If(w >= 8, [ir.Break()], None), ir.If(w == 5, [ir.Continue()], None), ir.TensorStore(output, [w], w)]
    synthetic = ir.For(w, integer(3), integer(5), integer(1), ir.ForKind.DEFAULT, body, {"synthetic_while": integer(1)})
    restored = apply("RestoreWhileLoops", tirx.Function([output], [synthetic]))
    assert len(objects(restored, ir.While)) == 1
    assert len([call for call in objects(restored, ir.Call) if call.op.name == "tirx.alloc_tensor"]) == 1
    assert not any(var.same_as(w) for var in objects(restored, ir.Var))
    executable = compile_native(restored, tmp_path)
    result = tvm_tirx.runtime.tensor(np.full(12, -1, dtype="int32"))
    executable(result)
    expected = np.full(12, -1, dtype="int32")
    expected[[3, 4, 6, 7]] = [3, 4, 6, 7]
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("step", [1, -1])
def test_loop_bound_snapshots_keep_evaluation_on_entry(step, tmp_path):
    bounds, output = tensor("bounds", (3,)), tensor("output", (8,))
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        ir.TensorLoad(bounds, [0]),
        ir.TensorLoad(bounds, [1]),
        ir.TensorLoad(bounds, [2]),
        ir.ForKind.DEFAULT,
        [ir.TensorStore(bounds, [1], integer(0)), ir.TensorStore(output, [i], i + 10)],
    )
    normalized = apply("NormalizeControlFlowForSchedule", tirx.Function([bounds, output], [root([loop])]))
    bind_values = [bind.value for bind in objects(normalized, ir.Bind)]
    assert len(bind_values) == 3 and all(isinstance(value, ir.TensorLoad) for value in bind_values)
    actual_loop = objects(normalized, ir.For)[0]
    assert all(isinstance(value, ir.Var) for value in (actual_loop.begin, actual_loop.end, actual_loop.step))
    executable = compile_native(normalized, tmp_path)
    begin, end = (1, 6) if step > 0 else (6, 1)
    state = tvm_tirx.runtime.tensor(np.array([begin, end, step], dtype="int32"))
    result = tvm_tirx.runtime.tensor(np.full(8, -1, dtype="int32"))
    executable(state, result)
    expected = np.full(8, -1, dtype="int32")
    # Native For always tests i < end; Python descending range is a frontend
    # lowering responsibility, and begin=6/end=1 is empty even for step=-1.
    if step > 0:
        for index in range(begin, end, step):
            expected[index] = index + 10
    np.testing.assert_array_equal(result.numpy(), expected)


def test_while_surrogate_is_multiple_of_all_version_counts():
    inner = ir.For(
        ir.Var("k", "int32"), integer(0), integer(3), integer(1), ir.ForKind.DEFAULT, [ir.Evaluate(0)], {"num_stages": integer(4)}
    )
    before = tirx.Function([], [root([ir.While(integer(True, "bool"), [inner, ir.Break()])])])
    after = apply("NormalizeControlFlowForSchedule", before)
    (synthetic,) = [loop for loop in objects(after, ir.For) if "synthetic_while" in loop.annotations]
    assert int(synthetic.end.value) == 12


def test_atomic_parallel_and_vf_scopes_preserve_while():
    while_loop = ir.While(integer(True, "bool"), [ir.Break()])
    parallel = ir.For(ir.Var("i", "int32"), integer(0), integer(2), integer(1), ir.ForKind.PARALLEL, [while_loop])
    vf = attribute("tl.sblock", {"name": "SIMD_VF", "no_realize": True}, [while_loop])
    after = apply("NormalizeControlFlowForSchedule", tirx.Function([], [root([parallel, vf])]))
    assert len(objects(after, ir.While)) == 2


def test_nested_same_name_iteration_vars_and_typed_shape_are_renamed():
    outer, inner = ir.Var("w", "int32"), ir.Var("w", "int32")
    shaped = tensor("shaped", (outer + 2,), scope="local")
    alloc = ir.Bind(shaped, ir.Call("tirx.alloc_tensor", [], ty_args=[shaped.ty]))
    nested = ir.For(inner, integer(0), integer(1), integer(1), ir.ForKind.DEFAULT, [ir.TensorStore(shaped, [inner], outer)])
    synthetic = ir.For(
        outer,
        integer(0),
        integer(2),
        integer(1),
        ir.ForKind.DEFAULT,
        [ir.If(outer >= 3, [ir.Break()], None), alloc, nested],
        {"synthetic_while": integer(1)},
    )
    result = apply("RestoreWhileLoops", tirx.Function([], [synthetic]))
    assert not any(var.same_as(outer) for var in objects(result, ir.Var))
    assert any(var.same_as(inner) for var in objects(result, ir.Var))
    shape_var = objects(result, tirx.TensorType)[-1].shape[0]
    assert shape_var is not None


def test_large_version_period_is_rejected_before_integer_overflow():
    loop = ir.For(ir.Var("i", "int32"), integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [ir.Break()], {"num_stages": integer(23)})
    before = tirx.Function([], [root([ir.While(integer(True, "bool"), [loop])])])
    with pytest.raises(ValueError, match="period exceeds int32"):
        apply("NormalizeControlFlowForSchedule", before)


def test_native_negative_step_with_less_than_condition_and_break(tmp_path):
    bounds, output = tensor("bounds", (3,)), tensor("output", (8,))
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        ir.TensorLoad(bounds, [0]),
        ir.TensorLoad(bounds, [1]),
        ir.TensorLoad(bounds, [2]),
        ir.ForKind.DEFAULT,
        [ir.If(i < 0, [ir.Break()], None), ir.TensorStore(bounds, [1], integer(-100)), ir.TensorStore(output, [i], i + 10)],
    )
    normalized = apply("NormalizeControlFlowForSchedule", tirx.Function([bounds, output], [root([loop])]))
    executable = compile_native(normalized, tmp_path)
    result = tvm_tirx.runtime.tensor(np.full(8, -1, dtype="int32"))
    executable(tvm_tirx.runtime.tensor(np.array([3, 5, -1], dtype="int32")), result)
    np.testing.assert_array_equal(result.numpy(), np.array([10, 11, 12, 13, -1, -1, -1, -1], dtype="int32"))
