import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx

from tilelang._compat import ir, stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("ascend_codegen")


def _nodes(root, kind):
    nodes = []
    stmt_functor.post_order_visit(root, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def _scope(name, body, **attrs):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name, **attrs}, body)


def _apply(name, body, parameters=(), config=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    assert ir.check_well_formed(function)
    with ir.transform.PassContext(config=config or {}):
        result = tvm_ffi.get_global_func("tl.transform." + name)()(ir.IRModule({"main": function}))["main"]
    assert ir.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert ir.check_well_formed(restored)
    return result


def _call_names(function):
    return [call.op.name for call in _nodes(function, ir.Call)]


def test_normalize_and_restore_while_with_guard_and_stage_period():
    flag = ir.Var("flag", "int32")
    index = ir.Var("index", "int32")
    nested = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, [], {"num_stages": 3})
    body = _scope("tilelang_root", ir.While(flag > 0, nested))
    normalized = _apply("NormalizeControlFlowForSchedule", body, [flag])
    synthetic = [loop for loop in _nodes(normalized, ir.For) if loop.annotations.get("synthetic_while")]
    assert len(synthetic) == 1 and int(synthetic[0].end) == 6
    assert len(_nodes(synthetic[0].body, ir.Break)) == 1
    assert isinstance(_nodes(synthetic[0].body, ir.If)[0].condition, ir.Var)
    restored = _apply("RestoreWhileLoops", normalized.body, [flag])
    assert len(_nodes(restored, ir.While)) == 1
    assert not any(loop.annotations.get("synthetic_while") for loop in _nodes(restored, ir.For))
    assert not any(binding.var.name.endswith("_counter") for binding in _nodes(restored, ir.Bind))


@pytest.mark.parametrize("opaque", ["SIMT_VF", "SIMD_VF", "user_scope"])
def test_control_normalizer_keeps_opaque_region_bodies(opaque):
    body = ir.While(ir.const(True), ir.Break())
    result = _apply("NormalizeControlFlowForSchedule", _scope("tilelang_root", _scope(opaque, body)))
    assert len(_nodes(result, ir.While)) == 1
    assert not _nodes(result, ir.For)


def test_control_normalizer_snapshots_mutable_loop_bounds():
    bounds = TensorView.create("bounds", [3], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, bounds.load(0), bounds.load(1), bounds.load(2), ir.ForKind.DEFAULT, bounds.store(1, 0))
    result = _apply("NormalizeControlFlowForSchedule", _scope("tilelang_root", loop), [bounds.tensor])
    snapshots = _nodes(result, ir.Bind)
    assert len(snapshots) == 3
    lowered = _nodes(result, ir.For)[0]
    assert all(bound.same_as(binding.var) for bound, binding in zip([lowered.begin, lowered.end, lowered.step], snapshots))


def test_while_counter_survives_versions_and_increments_before_continue():
    output = TensorView.create("output", [8], "int32")
    counter = ir.Var("counter", "int32")
    inner = ir.Var("inner", "int32")
    body = [
        output.store(counter % 8, counter),
        ir.For(inner, 0, 2, 1, ir.ForKind.DEFAULT, ir.Continue()),
        ir.If(counter < 4, ir.Continue(), None),
        ir.Break(),
    ]
    loop = ir.For(counter, 0, 8, 1, ir.ForKind.DEFAULT, body, {"synthetic_while": 1})
    result = _apply("RestoreWhileLoops", loop, [output.tensor])
    allocations = [binding for binding in _nodes(result, ir.Bind) if binding.var.name == "counter_counter"]
    assert len(allocations) == 1
    counter_tensor = allocations[0].var
    increments = [store for store in _nodes(result, ir.TensorStore) if store.dest.same_as(counter_tensor)]
    assert len(increments) == 2
    conditional = _nodes(result, ir.If)[0]
    assert isinstance(conditional.then_case.seq[-1], ir.Continue)
    assert isinstance(conditional.then_case.seq[-2], ir.TensorStore)
    assert _nodes(result, ir.While)[0].body.seq[-1].same_as(conditional.then_case.seq[-2])
    assert len(_nodes(_nodes(result, ir.For)[0], ir.TensorStore)) == 0


@pytest.mark.parametrize("begin,end,step", [(0, 8, 2), (8, -1, -3), (3, 3, -2), (0, 3, -2)])
def test_explicit_unroll_preserves_step_and_native_ssa(begin, end, step, compile_tilelang):
    output = TensorView.create("output", [16], "int32")
    local = TensorView.create("local", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.UNROLLED,
        [local.allocate(), local.store(0, index), output.store((index - begin) // step, local.load(0))],
        {"pragma_unroll_explicit": True},
    )
    result = _apply("UnrollLoopSkipVF", loop, [output.tensor])
    assert not _nodes(result, ir.For)
    definitions = [binding.var for binding in _nodes(result, ir.Bind)]
    assert len({hash(variable) for variable in definitions}) == len(range(begin, end, step))
    executable = compile_tilelang(result, target="c")
    actual = runtime.tensor(np.full(16, -99, dtype="int32"))
    executable(actual)
    expected = np.full(16, -99, dtype="int32")
    values = np.arange(begin, end, step, dtype="int32")
    expected[: len(values)] = values
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("name", ["SIMT_VF", "SIMD_VF"])
def test_explicit_unroll_preserves_vf_loops(name):
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.UNROLLED, [], {"pragma_unroll_explicit": True})
    result = _apply("UnrollLoopSkipVF", _scope(name, loop))
    assert len(_nodes(result, ir.For)) == 1


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_explicit_unroll_rejects_targeted_control(control):
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.UNROLLED, control(), {"pragma_unroll_explicit": True})
    with pytest.raises(ValueError, match="break or continue"):
        _apply("UnrollLoopSkipVF", loop)


def test_dcache_bypass_tracks_native_aliases_and_preserves_vf():
    storage = TensorView.create("storage", [4], "float32")
    alias = TensorView.create("alias", [4], "float32")
    readonly = TensorView.create("readonly", [4], "float32")
    body = [
        alias.declare(tirx.tensor_data_ptr(storage.tensor)),
        alias.store(0, readonly.load(0)),
        ir.Evaluate(storage.load(1)),
        _scope("SIMT_VF", ir.Evaluate(alias.load(2))),
    ]
    result = _apply("MarkScalarDcacheBypass", body, [storage.tensor, readonly.tensor])
    names = _call_names(result)
    assert names.count("tl.ascend_write_gm_bypass_dcache") == 1
    assert names.count("tl.ascend_read_gm_bypass_dcache") == 1
    assert "tl.ascend_read_gm_bypass_dcache" not in _call_names(_nodes(result, ir.RegionStmt)[0])
    assert any(load.source.same_as(readonly.tensor) for load in _nodes(result, ir.TensorLoad))


def test_simd_merging_uses_explicit_read_write_destination():
    destination = TensorView.create("destination", [1], "float32", scope="local.var")
    call = ir.Call("tl.simd.vadd", [destination.load(0), ir.const(1.0), "MODE_MERGING"], ty="float32")
    result = _apply("LegalizeSimdMerging", [destination.allocate(), destination.store(0, call)])
    merging = [call for call in _nodes(result, ir.Call) if call.op.name == "tl.simd.vadd"][0]
    assert not _nodes(result, ir.TensorStore)
    assert merging.ty == ir.PrimType("void")
    assert merging.args[0].op.name == "tl.access_ptr"
    assert int(merging.args[0].args[2]) == 3


def test_simd_merging_rejects_unassigned_expression():
    call = ir.Call("tl.simd.vadd", [ir.const(1.0), ir.const(2.0), "MODE_MERGING"], ty="float32")
    with pytest.raises(ValueError, match="requires assignment"):
        _apply("LegalizeSimdMerging", ir.Evaluate(call))


@pytest.mark.parametrize("count", [1, 3])
def test_manual_buffer_versions_become_native_region_metadata(count):
    tensor = TensorView.create("storage", [4], "float32", scope="shared")
    body = _scope("tilelang_root", [tensor.allocate()], **{"tl.manual_multi_buffer": {tensor.tensor: count}})
    result = _apply("NormalizeBufferVersion", body)
    root = _nodes(result, ir.RegionStmt)[0]
    assert "tl.manual_multi_buffer" not in root.attrs
    assert int(root.attrs["tl.buffer_version"][tensor.tensor]) == count
    restored = ir.load_json(ir.save_json(result))
    restored_root = _nodes(restored, ir.RegionStmt)[0]
    storage = _nodes(restored_root, ir.Bind)[0].var
    assert int(restored_root.attrs["tl.buffer_version"][storage]) == count
    renewed = ir.renew_def(result)
    renewed_root = _nodes(renewed, ir.RegionStmt)[0]
    new_storage = _nodes(renewed_root, ir.Bind)[0].var
    assert not new_storage.same_as(tensor.tensor)
    assert int(renewed_root.attrs["tl.buffer_version"][new_storage]) == count
    ir.assert_structural_equal(result, renewed)


@pytest.mark.parametrize("annotation,count", [("tl.manual_multi_buffer", 0), ("tl.manual_multi_buffer", -1), ("tl.buffer_version", 2)])
def test_manual_buffer_versions_reject_invalid_or_internal_input(annotation, count):
    tensor = TensorView.create("storage", [4], "float32", scope="shared")
    body = _scope("tilelang_root", [tensor.allocate()], **{annotation: {tensor.tensor: count}})
    with pytest.raises(ValueError, match="positive|compiler-internal"):
        _apply("NormalizeBufferVersion", body)


def _flag(name, event):
    return ir.Evaluate(ir.Call("tl.ascend_" + name + "_flag", ["MTE2_MTE3", event], ty="void"))


def test_flag_slots_compact_sparse_native_ids():
    result = _apply("RewriteFlagToBuf", [_flag("set", 17), _flag("wait", 17)])
    calls = _nodes(result, ir.Call)
    assert [int(call.args[1]) for call in calls] == [0, 0]
    assert [call.op.name for call in calls] == ["tl.ascend_set_flag", "tl.ascend_wait_flag"]


def test_flag_ranges_spill_as_indivisible_blocks():
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 12, 1, ir.ForKind.DEFAULT, [_flag("set", index), _flag("wait", index)])
    result = _apply("RewriteFlagToBuf", loop)
    calls = _nodes(result, ir.Call)
    assert [call.op.name for call in calls] == ["tl.ascend_get_buf", "tl.ascend_rls_buf"] * 2
    assert [str(call.args[0].value) for call in calls] == ["PIPE_MTE2", "PIPE_MTE2", "PIPE_MTE3", "PIPE_MTE3"]
    assert [int(call.args[2]) for call in calls] == [1, 1, 0, 0]


def test_flag_ranges_reject_mutex_pool_overflow():
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 33, 1, ir.ForKind.DEFAULT, _flag("set", index))
    with pytest.raises(Exception, match="exceed the shared pool"):
        _apply("RewriteFlagToBuf", loop)


@pytest.mark.parametrize("manual", [False, True])
def test_dual_copy_uses_native_vector_index(manual):
    source = TensorView.create("source", [4, 16], "float32")
    destination = TensorView.create("destination", [2, 16], "float32", scope="shared")
    copy = ir.Evaluate(ir.Call("tl.tileop.ascend_copy", [source.tensor, destination.tensor], {"dual_dst_ctl": 1}))
    body = [destination.allocate(), copy]
    if manual:
        sid = ir.Var("sid", "int32")
        body = _scope("VECTOR", ir.RegionStmt("tl.launch_grid", ["cthread", 2], [sid], {}, body))
    result = _apply("RewriteDualCopy", _scope("tilelang_root", body), [source.tensor], {"tl.enable_auto_schedule": not manual})
    launches = [region for region in _nodes(result, ir.RegionStmt) if region.op.name == "tl.launch_grid"]
    assert len(launches) == 1 and int(launches[0].args[1]) == 2
    calls = [call for call in _nodes(result, ir.Call) if call.op.name == "tl.tileop.ascend_copy"]
    assert len(calls) == 1
    assert "dual_dst_ctl" not in calls[0].attrs
    operation = tvm_ffi.get_global_func("tl.ParseOperator")(calls[0], {})
    assert int(operation.src_range[0].extent) == 2
    from tvm_tirx import sym

    assert sym.Analyzer().can_prove_equal(operation.src_range[0].begin, launches[0].body_params[0] * 2)


def test_dual_copy_rejects_missing_manual_vector_scope():
    source = TensorView.create("source", [4, 16], "float32")
    destination = TensorView.create("destination", [2, 16], "float32", scope="shared")
    copy = ir.Evaluate(ir.Call("tl.tileop.ascend_copy", [source.tensor, destination.tensor], {"dual_dst_ctl": 1}))
    with pytest.raises(ValueError, match="explicit T.Vector"):
        _apply(
            "RewriteDualCopy", _scope("tilelang_root", [destination.allocate(), copy]), [source.tensor], {"tl.enable_auto_schedule": False}
        )


def test_ascend_noop_removal_keeps_bindings_and_drops_dead_storage():
    value = ir.Var("value", "int32")
    unused = TensorView.create("unused", [4], "int32", scope="local")
    result = _apply("AscendRemoveNoOp", [ir.Bind(value, ir.const(3)), unused.allocate(), ir.Evaluate(0)])
    assert [binding.var.name for binding in _nodes(result, ir.Bind)] == ["value"]


def test_ascend_noop_dataflow_does_not_leak_between_branches(compile_tilelang):
    storage = TensorView.create("storage", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    flag = ir.Var("flag", "int32")
    body = ir.If(flag > 0, storage.store(0, 1), output.store(0, storage.load(0)))
    result = _apply(
        "AscendRemoveNoOp", body, [storage.tensor, output.tensor, flag], {"tl.AscendRemoveNoOp": {"use_dataflow_analysis": True}}
    )
    executable = compile_tilelang(result, target="c")
    data = runtime.tensor(np.array([7], dtype="int32"))
    actual = runtime.tensor(np.array([-1], dtype="int32"))
    executable(data, actual, 0)
    np.testing.assert_array_equal(actual.numpy(), [7])


def test_ascend_noop_dataflow_eliminates_redundant_stores():
    storage = TensorView.create("storage", [1], "int32")
    result = _apply(
        "AscendRemoveNoOp",
        [storage.store(0, 3), storage.store(0, 3)],
        [storage.tensor],
        {"tl.AscendRemoveNoOp": {"use_dataflow_analysis": True}},
    )
    assert len(_nodes(result, ir.TensorStore)) == 1


def _hint(first, second, level, cross=0, group="", conflict=False):
    return ir.Evaluate(ir.Call("tl.conflict_hint", [first, second, level, cross, group, conflict], ty="void"))


def test_conflict_hint_attaches_to_native_loop_and_root():
    first = TensorView.create("first", [4], "int32")
    second = TensorView.create("second", [4], "int32")
    index = ir.Var("index", "int32")
    body = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, _hint(first.tensor, second.tensor, -2, 1, conflict=True))
    result = _apply("NormalizeConflictHints", _scope("tilelang_root", body), [first.tensor, second.tensor])
    assert "tl.conflict_hint" not in _call_names(result)
    hint = _nodes(result, ir.For)[0].annotations["conflict_hint"][0]
    assert hint[0].same_as(first.tensor) and hint[1].same_as(second.tensor)
    assert int(hint[2]) == 1 and bool(hint[3])
    assert len(_nodes(result, ir.RegionStmt)[0].attrs["tl.root_conflict_hints"]) == 1


def test_grouped_conflict_hint_uses_common_enclosing_loop():
    first = TensorView.create("first", [4], "int32")
    second = TensorView.create("second", [4], "int32")
    outer, left, right = [ir.Var(name, "int32") for name in ("outer", "left", "right")]
    siblings = [
        ir.For(left, 0, 4, 1, ir.ForKind.DEFAULT, _hint(first.tensor, first.tensor, 0, group="pair")),
        ir.For(right, 0, 4, 1, ir.ForKind.DEFAULT, _hint(second.tensor, second.tensor, 0, group="pair")),
    ]
    body = ir.For(outer, 0, 4, 1, ir.ForKind.DEFAULT, siblings)
    result = _apply("NormalizeConflictHints", _scope("tilelang_root", body), [first.tensor, second.tensor])
    annotated = [loop for loop in _nodes(result, ir.For) if "conflict_hint" in loop.annotations]
    assert len(annotated) == 1 and annotated[0].var.name == "outer"


def test_grouped_conflict_hint_rejects_unpaired_marker():
    tensor = TensorView.create("tensor", [4], "int32")
    body = _scope("tilelang_root", _hint(tensor.tensor, tensor.tensor, -1, group="missing"))
    with pytest.raises(Exception, match="appears only once"):
        _apply("NormalizeConflictHints", body, [tensor.tensor])
