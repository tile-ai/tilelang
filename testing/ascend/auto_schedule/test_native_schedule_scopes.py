import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang.ascend.language as T
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _scope(key, payload, body):
    return ir.RegionStmt("tl.scope", [], [], {key: payload}, body)


def _module(body, params=(), attrs=None):
    root = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root", **(attrs or {})}, body)
    function = tirx.Function(list(params), root, ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target("ascend")).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    return ir.IRModule({"main": function})


def _materialize(module):
    result = tvm_ffi.get_global_func("tl.transform.MaterializeScheduleUnits")()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _markers(module, key):
    return [region for region in _nodes(module, ir.RegionStmt) if key in region.attrs.dict]


def _root_units(module):
    root = next(region for region in _nodes(module, ir.RegionStmt) if region.attrs.dict.get("name_hint") == "tilelang_root")
    return list(root.body.seq)


def test_scalar_results_are_lexical_and_available_to_later_tasks():
    source = ir.Var("source", "int32")
    first = ir.Var("first", "int32")
    second = ir.Var("second", "int32")
    module = _module([ir.Bind(first, source + 1), ir.Bind(second, first * 2), ir.Evaluate(second)], [source])
    result = _materialize(module)
    units = _root_units(result)
    assert len(units) == 3
    assert units[0].result_vars[0].same_as(first)
    assert units[1].result_vars[0].same_as(second)
    assert units[0].attrs.dict["tl.schedule_yield"][0].same_as(first) is False
    bindings = _nodes(result, ir.Bind)
    assert all(not binding.var.same_as(first) and not binding.var.same_as(second) for binding in bindings)
    assert all(region.op.name == "tl.ascend.schedule_scope" for region in _markers(result, "tl.schedule_unit"))


def test_dependent_tensor_result_types_remain_well_formed():
    extent = ir.Var("extent", "int32")
    size = ir.Var("size", "int32")
    tensor = TensorView.create("tensor", [size], "float32", scope="shared.dyn")
    module = _module([ir.Bind(size, extent + 1), tensor.allocate(), tensor.store(0, ir.const(2.0))], [extent])
    result = _materialize(module)
    units = _root_units(result)
    assert units[0].result_vars[0].same_as(size)
    assert units[1].result_vars[0].same_as(tensor.tensor)


def test_equivalent_nonidentical_binding_types_preserve_exported_identity():
    tensor = TensorView.create("tensor", [32], "float32", scope="shared.dyn")
    allocation_type = ir.load_json(ir.save_json(tensor.tensor.ty))
    allocation = ir.Call("tirx.alloc_tensor", [], ty_args=[allocation_type])
    initial = _module([ir.Bind(tensor.tensor, allocation), tensor.store(0, ir.const(1.0))])
    materialized = _materialize(initial)
    assert _root_units(materialized)[0].result_vars[0].same_as(tensor.tensor)
    result = _roundtrip(materialized)
    assert _root_units(result)[0].result_vars[0].same_as(tensor.tensor)


def test_guarded_bind_results_are_not_evaluated_as_region_arguments():
    predicate = ir.Var("predicate", "bool")
    source = TensorView.create("source", [1], "float32")
    value = ir.Var("value", "float32")
    conditional = ir.If(predicate, [ir.Bind(value, source.load(0)), ir.Evaluate(value + ir.const(1.0))], None)
    result = _materialize(_module(conditional, [predicate, source.tensor]))
    guards = _markers(result, "tl.schedule_condition")
    assert len(guards) == 2
    assert all(not guard.args for guard in guards)
    assert len(_nodes(guards[0].body, ir.TensorLoad)) + len(_nodes(guards[1].body, ir.TensorLoad)) == 1
    assert _root_units(result)[0].result_vars[0].same_as(value)


def test_stage_normalization_and_nonzero_signed_loops():
    index = ir.Var("index", "int32")
    body = [_scope("tl.ascend_stage", 2, ir.Evaluate(index)), ir.Evaluate(index + 1)]
    loop = ir.For(index, 9, -3, -2, ir.ForKind.DEFAULT, body)
    result = _materialize(_module(loop))
    loops = _nodes(result, ir.For)
    assert len(loops) == 1
    assert [int(loops[0].begin), int(loops[0].end), int(loops[0].step)] == [9, -3, -2]
    assert [int(unit.attrs.dict["tl.schedule_unit"]["stage"]) for unit in loops[0].body.seq] == [2, 0]


def test_else_guards_and_loop_local_variables_stay_local():
    predicate = ir.Var("predicate", "bool")
    index = ir.Var("index", "int32")
    local = ir.Var("local", "int32")
    loop = ir.For(index, 2, 6, 1, ir.ForKind.DEFAULT, [ir.Bind(local, index * 2), ir.Evaluate(local)])
    conditional = ir.If(predicate, loop, ir.Evaluate(ir.const(4)))
    result = _materialize(_module(conditional, [predicate]))
    assert all(not unit.result_vars for unit in _root_units(result))
    assert len(_markers(result, "tl.schedule_condition")) == 2


@pytest.mark.parametrize("marker", ["tl.ascend_task", "tl.ascend_per_core_task", "tl.ascend_stage"])
def test_rejects_nested_task_and_stage_markers(marker):
    nested = _scope(marker, 1 if marker == "tl.ascend_stage" else {}, ir.Evaluate(ir.const(0)))
    module = _module(_scope("tl.ascend_task", {}, nested))
    with pytest.raises(Exception, match="nested|Nested|complete scheduler task"):
        _materialize(module)


def test_rejects_multiple_pipes_in_explicit_task():
    source = TensorView.create("source", [64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    output = TensorView.create("output", [64], "float32")
    task = _scope(
        "tl.ascend_task", {}, [ir.Evaluate(T.copy(source.tensor, shared.tensor)), ir.Evaluate(T.copy(shared.tensor, output.tensor))]
    )
    with pytest.raises(Exception, match="multiple pipes"):
        _materialize(_module([shared.allocate(), task], [source.tensor, output.tensor]))


def test_per_core_candidates_allow_dependency_free_scalar_control():
    source = TensorView.create("source", [64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    block = ir.Var("block", "int32")
    parity = ir.Var("parity", "int32")
    copy = ir.Evaluate(T.copy(source.tensor, shared.tensor))
    per_core = _scope("tl.ascend_per_core_task", {}, [ir.Bind(parity, block % 2), ir.If(parity == 0, copy, copy)])
    result = _materialize(_module([shared.allocate(), per_core], [source.tensor, block]))
    assert len(_markers(result, "tl.ascend_per_core_task")) == 1
    assert len(_markers(result, "tl.ascend_task")) == 4


def test_per_core_unmarked_scalar_buffer_access_is_rejected():
    source = TensorView.create("source", [1], "int32")
    value = ir.Var("value", "int32")
    per_core = _scope("tl.ascend_per_core_task", {}, ir.Bind(value, source.load(0)))
    with pytest.raises(Exception, match="unmarked scalar Bind.*may not access buffers"):
        _materialize(_module(per_core, [source.tensor]))


def test_frontend_task_stage_and_per_core_use_native_regions():
    @T.prim_func
    def main(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((64,), "float32")
            with T.Stage(0), T.Task(latency=5, ii=2):
                T.copy(source, shared)
            with T.Stage(1), T.PerCoreTask():
                T.copy(shared, output)

    result = _materialize(ir.IRModule({"main": main.with_attr("target", Target("ascend"))}))
    assert len(_markers(result, "tl.ascend_per_core_task")) == 1
    tasks = _markers(result, "tl.ascend_task")
    assert any(int(task.attrs.dict["tl.ascend_task"].get("latency", 0)) == 5 for task in tasks)
    assert not _markers(result, "tl.ascend_stage")


def test_cannot_materialize_twice():
    result = _materialize(_module(ir.Evaluate(ir.const(0))))
    with pytest.raises(Exception, match="already-materialized"):
        _materialize(result)


def _roundtrip(module):
    function = tvm_ffi.get_global_func("tl.ascend.RoundTripScheduledTIR")(module["main"])
    result = ir.IRModule({"main": function})
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def test_schedule_codec_preserves_guarded_scalar_and_loop_scopes():
    predicate = ir.Var("predicate", "bool")
    index = ir.Var("index", "int32")
    value = ir.Var("value", "int32")
    loop = ir.For(index, 8, -2, -2, ir.ForKind.DEFAULT, [ir.Bind(value, index + 1), ir.If(predicate, ir.Evaluate(value), None)])
    initial = _materialize(_module(loop, [predicate]))
    result = initial
    for _ in range(3):
        result = _roundtrip(result)
        ir.assert_structural_equal(initial, result)


def test_schedule_codec_preserves_kernel_storage_metadata():
    first = TensorView.create("first", [32], "float32", scope="shared.dyn")
    second = TensorView.create("second", [32], "float32", scope="shared.dyn")
    attrs = {
        "tl.buffer_versions_map": {first.tensor: 2},
        "tl.manual_multi_buffer": {second.tensor: 3},
        "tl.buffer_version_mode": {first.tensor: "counter"},
        "tl.unlimit_memory_scopes": ["shared.dyn"],
        "tl.buffer_alias_map": {first.tensor: [second.tensor], second.tensor: [first.tensor]},
        "vector_count": 2,
        "custom_metadata": "preserved",
    }
    initial = _materialize(_module([first.allocate(), second.allocate(), ir.Evaluate(T.fill(first.tensor, 0.0))], attrs=attrs))
    result = _roundtrip(initial)
    ir.assert_structural_equal(initial, result)


def test_schedule_codec_rejects_multiple_tasks_in_one_unit():
    tasks = [_scope("tl.ascend_task", {}, ir.Evaluate(ir.const(value))) for value in [1, 2]]
    module = _module(_scope("tl.schedule_unit", {"stage": 0}, tasks))
    with pytest.raises(Exception, match="exactly one outer T.Task"):
        _roundtrip(module)


def test_schedule_codec_rejects_missing_schedule_units():
    value = ir.Var("value", "int32")
    with pytest.raises(Exception, match="MaterializeScheduleUnits"):
        _roundtrip(_module(ir.Evaluate(value), [value]))


def test_schedule_codec_retains_task_cost_and_per_core_contract():
    source = TensorView.create("source", [64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    task = _scope("tl.ascend_per_core_task", {"latency": 20, "ii": 4}, ir.Evaluate(T.copy(source.tensor, shared.tensor)))
    initial = _materialize(_module([shared.allocate(), task], [source.tensor]))
    result = _roundtrip(initial)
    ir.assert_structural_equal(initial, result)
    metadata = _markers(result, "tl.ascend_per_core_task")[0].attrs.dict["tl.ascend_per_core_task"]
    assert [int(metadata["latency"]), int(metadata["ii"])] == [20, 4]
