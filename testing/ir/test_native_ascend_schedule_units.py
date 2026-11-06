"""Native Ascend task boundaries preserve lexical evaluation and SSA definitions."""

from testing.ascend._ir import kernel_body

import os
from pathlib import Path

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T
from testing.ascend._block import SEMANTIC_SCOPE_OPS, attribute_fields, has_attribute
from testing.ascend._ir import scheduled_fields

from test_native_ascend_dual_copy import _materialize
from test_native_ascend_ops import function, strided_tensor
from test_native_cuda_copy import calls
from test_native_reducer_contracts import (
    allocate,
    annotation,
    attribute,
    integer,
    objects,
    region,
    run_pass,
    tensor,
    tileop,
)

SCHEDULED_BIND = "tl.ascend.scheduled_bind"


def _fields(node):
    if isinstance(node, ir.RegionStmt) and node.op.name in SEMANTIC_SCOPE_OPS:
        return attribute_fields(node)
    # This shared fixture also reads tl.annotation Calls, including direct
    # conflict-hint carriers that are not lexical regions.
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])


def _attributes(value, key):
    return [node for node in objects(value, ir.RegionStmt) if has_attribute(node, key)]


def _root(body):
    return ir.SeqStmt(body)


def _kernel(params, body):
    return function(params, [_root(body)])


def _task(body, metadata=None):
    return attribute(metadata or {}, "tl.ascend_task", integer(1), body)


def _stage(stage, body):
    return attribute(integer(stage, "int64"), "tl.ascend_stage", integer(1), body)


def _copy(source, destination):
    return tileop("ascend_copy", region(source), region(destination))


def _bind_target(value):
    return value.with_attr("target", tvm_tirx.target.Target("ascend"))


def _artifact(name, value):
    directory = os.environ.get("TILELANG_OPS_ARTIFACT_DIR")
    if directory:
        (Path(directory) / (name + ".json")).write_text(ir.save_json(value))


def _check(value, name=None):
    before = ir.save_json(value)
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    result = run_pass("MaterializeScheduleUnits", value)
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    roundtrip = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, roundtrip)
    assert tirx.analysis.check_well_formed(roundtrip)
    assert not tirx.analysis.undefined_vars(roundtrip)
    repeated = run_pass("MaterializeScheduleUnits", result)
    ir.assert_structural_equal(result, repeated)
    assert ir.save_json(result) == ir.save_json(repeated)
    if name:
        _artifact(name, result)
    return result


def _scheduled_bind(expression, *, stage=-1, task=None, metadata=None, attrs=None):
    fields = {"stage": integer(stage, "int64"), "task": task or {}} if metadata is None else metadata
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    attributes = {"schema_version": 1, "operand_count": 1, "annotation_schema": schema}
    attributes.update(attrs or {})
    return ir.Call(SCHEDULED_BIND, [expression, payload], attrs=attributes)


def _binding(value, variable):
    return next(bind for bind in objects(value, ir.Bind) if bind.var.same_as(variable))


def _stages(body):
    result = []
    for statement in body.seq:
        if isinstance(statement, ir.Bind) and isinstance(statement.value, ir.Call) and statement.value.op.name == SCHEDULED_BIND:
            result.append(int(scheduled_fields(statement)["stage"]))
        elif isinstance(statement, ir.RegionStmt) and statement.op.name in SEMANTIC_SCOPE_OPS:
            fields = _fields(statement)
            if fields["key"] == "tl.schedule_unit":
                result.append(int(fields["node"]["stage"]))
            elif fields["key"] in {"tilelang_assume", "tl.assume_requires_runtime_check", "tl.loop.annotations"}:
                result.extend(_stages(statement.body))
        elif isinstance(statement, ir.If):
            result.extend(_stages(statement.then_case))
            if statement.else_case is not None:
                result.extend(_stages(statement.else_case))
    return result


def test_lexical_scope_preserves_local_definitions_through_scheduling():
    output = tensor("output", [4], "int32", "shared.dyn")
    local = ir.Var("local", "int32")
    scope = ir.RegionStmt(
        "tl.lexical_scope",
        [],
        [],
        {},
        [ir.Bind(local, integer(3)), ir.TensorStore(output, [integer(0)], local)],
    )
    result = _check(_kernel([output], [_stage(0, [scope])]))
    for name in ("AssignCore", "ResolveCore", "InsertSync", "LowerScheduledTIR"):
        result = run_pass(name, result)
        assert tirx.analysis.check_well_formed(result)
        assert not tirx.analysis.undefined_vars(result)
    assert any(node.op.name == "tl.lexical_scope" for node in objects(result, ir.RegionStmt))
    (store,) = objects(result, ir.TensorStore)
    (binding,) = [node for node in objects(result, ir.Bind) if node.var.name == "local"]
    assert store.value.same_as(binding.var)
    assert int(binding.value) == 3


def test_flat_bind_keeps_scalar_shape_stride_origin_and_metadata_def_use():
    incoming = ir.Var("extent", "int32")
    length = ir.Var("extent", "int32")
    index, loaded = ir.Var("index", "int32"), ir.Var("loaded", "int32")
    output = tensor("output", [1], "int32", "global")
    scratch = strided_tensor("scratch", [length], [2], 7, "int32", "shared.dyn")
    source = length - 1
    declaration = allocate(scratch)
    metadata = annotation({"keep": {scratch: [length, incoming]}})
    value = _kernel(
        [incoming, output],
        [
            ir.Bind(length, incoming + 1),
            declaration,
            metadata,
            ir.Bind(index, source),
            ir.TensorStore(scratch, [index], integer(9)),
            ir.Bind(loaded, ir.TensorLoad(scratch, [index])),
            ir.TensorStore(output, [integer(0)], loaded),
        ],
    )
    result = _check(value, "native-schedule-scalar-shape-def-use")
    assert len(calls(result, SCHEDULED_BIND)) == 4
    for variable in (length, scratch, index, loaded):
        original, rewritten = _binding(value, variable), _binding(result, variable)
        fields = scheduled_fields(rewritten)
        assert rewritten.value.args[0].same_as(original.value)
        assert rewritten.var.same_as(original.var)
        ir.assert_structural_equal(rewritten.value.ty, original.value.ty)
        assert int(fields["stage"]) == -1 and len(fields["task"]) == 0
    body = kernel_body(result)
    assert body.seq[2].same_as(metadata)
    assert scratch.ty.shape[0].same_as(length)
    assert not length.same_as(incoming)
    ir.assert_structural_equal(_binding(result, scratch).value.ty.layout, scratch.ty.layout)


def test_old_flatten_is_invalid_for_native_bind_but_lexical_schedule_is_valid():
    enabled = ir.Var("enabled", "bool")
    output = tensor("output", [1], "int32", "global")
    local = ir.Var("branch_value", "int32")
    binding = ir.Bind(local, integer(11))
    store = ir.TensorStore(output, [integer(0)], local)
    good = _kernel([enabled, output], [ir.If(enabled, [binding, store], None)])
    # This is the old per-statement AttrStmt flatten shape, represented with
    # real zero-result native regions. The second task cannot see the first's
    # local Bind definition, even though both tasks have the same condition.
    bad = _kernel(
        [enabled, output],
        [
            attribute({"stage": integer(-1, "int64")}, "tl.schedule_unit", integer(1), [ir.If(enabled, [_task([binding])], None)]),
            attribute({"stage": integer(-1, "int64")}, "tl.schedule_unit", integer(1), [ir.If(enabled, [_task([store])], None)]),
        ],
    )
    _artifact("native-schedule-old-flatten-invalid", bad)
    assert not tirx.analysis.check_well_formed(bad, assert_mode=False)
    with pytest.raises((ValueError, RuntimeError, TypeError), match="defined|scope|variable|Variable"):
        run_pass("MaterializeScheduleUnits", bad)
    result = _check(good, "native-schedule-lexical-bind-counterexample")
    (branch,) = objects(result, ir.If)
    assert isinstance(branch.then_case.seq[0], ir.Bind)
    assert _binding(result, local).var.same_as(local)
    assert len(_attributes(branch.then_case, "tl.schedule_unit")) == 1


def test_same_named_variables_in_sibling_branches_keep_distinct_definitions():
    enabled = ir.Var("enabled", "bool")
    output = tensor("output", [2], "int32", "global")
    outer, lhs, rhs = (ir.Var("value", "int32") for _ in range(3))
    value = _kernel(
        [enabled, output],
        [
            ir.Bind(outer, integer(5)),
            ir.If(
                enabled,
                [ir.Bind(lhs, outer + 1), ir.TensorStore(output, [integer(0)], lhs)],
                [ir.Bind(rhs, outer + 2), ir.TensorStore(output, [integer(0)], rhs)],
            ),
            ir.TensorStore(output, [integer(1)], outer),
        ],
    )
    result = _check(value)
    assert len(calls(result, SCHEDULED_BIND)) == 3
    assert all(_binding(result, var).var.same_as(var) for var in (outer, lhs, rhs))
    stores = objects(result, ir.TensorStore)
    assert any(store.value.same_as(lhs) for store in stores)
    assert any(store.value.same_as(rhs) for store in stores)
    assert any(store.value.same_as(outer) for store in stores)


@pytest.mark.parametrize("wrapper", ["if", "tilelang_assume", "tl.assume_requires_runtime_check"])
def test_mutable_guard_is_evaluated_once_in_its_original_scope(wrapper):
    flag = tensor("flag", [1], "int32", "global")
    output = tensor("output", [1], "int32", "global")
    condition = ir.TensorLoad(flag, [integer(0)]) != 0
    local = ir.Var("captured", "int32")
    body = [ir.TensorStore(flag, [integer(0)], integer(0)), ir.Bind(local, integer(17)), ir.TensorStore(output, [integer(0)], local)]
    scope = ir.If(condition, body, None) if wrapper == "if" else attribute(condition, wrapper, integer(1), body)
    result = _check(_kernel([flag, output], [scope]))
    loads = [load for load in objects(result, ir.TensorLoad) if load.source.same_as(flag)]
    assert len(loads) == 1
    if wrapper == "if":
        (preserved,) = objects(result, ir.If)
        assert preserved.condition.same_as(scope.condition)
    else:
        (preserved,) = _attributes(result, wrapper)
        assert _fields(preserved)["node"].same_as(_fields(scope)["node"])
    assert len(preserved.then_case.seq if wrapper == "if" else preserved.body.seq) == 3


@pytest.mark.parametrize("kind", ["primitive", "pointer", "tensor"])
def test_scheduled_binding_infers_its_real_value_type(kind):
    ty = {"primitive": ir.PrimType("int32"), "pointer": ir.PtrType(ir.PrimType("float32")), "tensor": tirx.TensorType([4], "float32")}[kind]
    value = ir.Var("value", ty)
    wrapped = _scheduled_bind(value)
    ir.assert_structural_equal(wrapped.ty, value.ty)
    assert wrapped.args[0].same_as(value)
    assert tvm_ffi.get_global_func("tl.SideEffect")(wrapped) > tvm_ffi.get_global_func("tl.SideEffect")(value)


def test_scheduled_binding_preserves_effect_count_locations_and_unused_tasks():
    expression_loc = ir.SourceLoc(ir.SourceName("scheduled_bind.py"), 7, 3, 7, 14)
    statement_loc = ir.SourceLoc(ir.SourceName("scheduled_bind.py"), 7, 0, 7, 14)
    effect = ir.Call("tirx.call_extern", [ir.StringImm("next_value")], ty="int32", loc=expression_loc)
    value = ir.Var("unused_value", "int32")
    source = _kernel([], [ir.Bind(value, effect, loc=statement_loc)])
    result = _check(source)
    binding = _binding(result, value)
    assert binding.loc.same_as(statement_loc)
    assert binding.value.loc.same_as(expression_loc)
    assert binding.value.args[0].same_as(effect)
    simplified = tirx.transform.StmtSimplify()(ir.IRModule({"main": result}))["main"]
    assert len(calls(simplified, "tirx.call_extern")) == 1
    assert len(calls(simplified, SCHEDULED_BIND)) == 1


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"task": {}}, "stage|exactly"),
        ({"stage": integer(-1, "int64")}, "task|exactly"),
        ({"stage": integer(-2, "int64"), "task": {}}, "stage"),
        ({"stage": integer(1 << 32, "int64"), "task": {}}, "stage"),
        ({"stage": ir.Var("stage", "int32"), "task": {}}, "constant integer"),
        ({"stage": integer(True, "bool"), "task": {}}, "constant integer"),
        ({"stage": integer(-1, "int64"), "task": {"ii": integer(0)}}, "ii"),
        ({"stage": integer(-1, "int64"), "task": {"latency": integer(-1)}}, "latency"),
        ({"stage": integer(-1, "int64"), "task": {"core_mask": integer(8)}}, "core_mask"),
        ({"stage": integer(-1, "int64"), "task": {}, "extra": 1}, "exactly"),
    ],
)
def test_scheduled_binding_rejects_invalid_metadata(fields, message):
    with pytest.raises((ValueError, RuntimeError, TypeError), match=message):
        _scheduled_bind(integer(5), metadata=fields)


@pytest.mark.parametrize("attrs", [{"schema_version": 0}, {"operand_count": 0}, {"operand_count": 2}])
def test_scheduled_binding_rejects_invalid_schema(attrs):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _scheduled_bind(integer(5), attrs=attrs)


def test_scheduled_binding_rejects_a_forged_result_type_nested_marker_and_void():
    fields = {"stage": integer(-1, "int64"), "task": {}}
    forged = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("float32"), ir.Op.get(SCHEDULED_BIND), [integer(3)], fields, ir.UnknownLoc()
    )
    # Native Call construction intentionally permits raw invalid IR; explicit
    # validation and the task consumer must reject the forged type.
    with pytest.raises((ValueError, RuntimeError, TypeError), match="type"):
        forged.validate()
    invalid = _kernel([], [ir.Bind(ir.Var("forged", "float32"), forged)])
    with pytest.raises((ValueError, RuntimeError, TypeError), match="type"):
        run_pass("MaterializeScheduleUnits", invalid)
    with pytest.raises((ValueError, RuntimeError, TypeError), match="another scheduled binding"):
        _scheduled_bind(_scheduled_bind(integer(3)))
    for ty in (ir.PrimType("void"), ir.TupleType([])):
        with pytest.raises((ValueError, RuntimeError, TypeError), match="void"):
            _scheduled_bind(ir.Call("tirx.call_extern", [ir.StringImm("no_value")], ty=ty))
    with pytest.raises((ValueError, RuntimeError, TypeError), match="concrete value type"):
        _scheduled_bind(ir.Call("tirx.call_extern", [ir.StringImm("unknown_type")]))


@pytest.mark.parametrize("materialized", [False, True])
def test_real_dsl_partial_manual_stages_keep_costs_and_native_loop_annotations(materialized):
    @T.prim_func
    def before(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(4, annotations={"enable_offset": True}):
                with T.Stage(1), T.Task(latency=7, ii=2):
                    T.copy(temp, B[i * 16 : (i + 1) * 16])
                with T.Task(latency=3, ii=1):
                    T.copy(A[i * 16 : (i + 1) * 16], temp)

    value = _bind_target(before)
    if materialized:
        value = _materialize(value)
    result = _check(value)
    (loop,) = objects(result, ir.For)
    assert _stages(loop.body) == [1, 0]
    costs = [_fields(task)["node"] for task in _attributes(loop.body, "tl.ascend_task")]
    assert [(int(cost["latency"]), int(cost["ii"])) for cost in costs] == [(7, 2), (3, 1)]
    assert loop.annotations["enable_offset"]
    assert not _attributes(result, "tl.ascend_stage")


@pytest.mark.parametrize("kind", ["serial", "unrolled"])
def test_lexical_loop_metadata_and_nonunit_bounds_remain_native(kind):
    incoming = ir.Var("n", "int32")
    output = tensor("output", [64], "int32", "global")
    induction = ir.Var("i", "int32")
    local = ir.Var("local_index", "int32")
    metadata = {"multi_buffer_eligible": [output], "keep": {output: induction}}
    scope = attribute(
        None,
        "tl.loop.annotations",
        metadata,
        [ir.Bind(local, induction + 1), ir.TensorStore(output, [local], integer(4))],
    )
    loop = ir.For(
        induction,
        integer(2),
        incoming,
        integer(2),
        ir.ForKind.DEFAULT if kind == "serial" else ir.ForKind.UNROLLED,
        [scope],
        annotations={"enable_offset": True},
    )
    result = _check(_kernel([incoming, output], [loop]))
    (rewritten,) = objects(result, ir.For)
    assert rewritten.var.same_as(induction)
    assert int(rewritten.begin) == 2 and int(rewritten.step) == 2
    assert rewritten.end.same_as(incoming)
    (retained,) = _attributes(rewritten, "tl.loop.annotations")
    ir.assert_structural_equal(_fields(retained)["value"], _fields(scope)["value"])


@pytest.mark.parametrize("explicit,mixed", [(True, False), (False, False), (True, True)])
def test_real_per_core_candidates_keep_conditions_and_grouping(explicit, mixed):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32"), enabled: T.bool):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            with T.PerCoreTask():
                if enabled:
                    if explicit:
                        with T.Task():
                            T.copy(A, l1)
                    else:
                        T.copy(A, l1)
                if not enabled:
                    if explicit and not mixed:
                        with T.Task():
                            T.copy(B, l1)
                    else:
                        T.copy(B, l1)

    result = _check(_bind_target(before))
    (group,) = _attributes(result, "tl.ascend_per_core_task")
    assert len(_attributes(group, "tl.ascend_task")) == 2
    assert len(objects(group, ir.If)) == 2


def test_per_core_local_scalar_control_keeps_its_definition_in_the_group_scope():
    source, destination = tensor("A", [16], "float32", "global"), tensor("l1", [16], "float32", "shared.l1")
    block = ir.Var("block", "int32")
    selected = ir.Var("selected", "bool")
    group = attribute(
        {},
        "tl.ascend_per_core_task",
        integer(1),
        [ir.Bind(selected, block == 0), ir.If(selected, [_copy(source, destination)], [_copy(source, destination)])],
    )
    result = _check(_kernel([source, destination, block], [group]))
    (retained,) = _attributes(result, "tl.ascend_per_core_task")
    assert isinstance(retained.body.seq[0], ir.Bind)
    assert not calls(retained.body.seq[0], SCHEDULED_BIND)
    (condition,) = objects(retained, ir.If)
    assert condition.condition.same_as(selected)


@pytest.mark.parametrize(
    "case,message",
    [
        ("cross-pipe", "exactly one Ascend hardware pipe"),
        ("cross-core", "both AIC and AIV HBM paths"),
        ("scalar-fill", "exactly one Ascend hardware pipe"),
        ("vector-copy", "exactly one Ascend hardware pipe"),
        ("two-syncs", "must be the task's only statement"),
        ("nested-per-core", "cannot be nested inside"),
        ("nested-task", "Nested T.Task"),
        ("nested-stage", "complete scheduler task"),
    ],
)
def test_real_dsl_invalid_atomic_tasks_keep_original_diagnostics(case, message):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            l0 = T.alloc_l0a((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            other = T.alloc_shared((16, 16), "float32")
            with T.Task():
                if case == "cross-pipe":
                    T.copy(A, l1)
                    T.copy(l1, l0)
                elif case == "cross-core":
                    T.copy(A, l1)
                    T.copy(A, ub)
                elif case == "scalar-fill":
                    T.copy(A, ub)
                    T.fill(ub, 0)
                elif case == "vector-copy":
                    T.copy(A, ub)
                    T.copy(ub, other)
                elif case == "two-syncs":
                    T.ascend_sync_inter_arrive("PIPE_V", 3)
                    T.ascend_sync_inter_wait("PIPE_V", 3)
                elif case == "nested-per-core":
                    with T.PerCoreTask():
                        T.copy(A, ub)
                elif case == "nested-task":
                    with T.Task():
                        T.copy(A, ub)
                else:
                    with T.Stage(1):
                        T.copy(A, ub)

    with pytest.raises((ValueError, RuntimeError, TypeError), match=message):
        run_pass("MaterializeScheduleUnits", _bind_target(before))


@pytest.mark.parametrize(
    "case,message", [("buffer-bind", "may not access buffers"), ("no-access", "dependency-bearing"), ("wrong-sync", "inferred pipe")]
)
def test_per_core_control_validation(case, message):
    source, destination = tensor("A", [16], "int32", "global"), tensor("l1", [16], "int32", "shared.l1")
    variable = ir.Var("control", "int32")
    body = [ir.Bind(variable, ir.TensorLoad(source, [integer(0)]) if case == "buffer-bind" else integer(0))]
    if case != "no-access":
        body.append(_copy(source, destination))
    if case == "wrong-sync":
        body.append(ir.Evaluate(ir.Call("tl.ascend_cross_core_set_flag", [integer(2), ir.StringImm("PIPE_V"), integer(3)], ty="void")))
    value = _kernel([source, destination], [attribute({}, "tl.ascend_per_core_task", integer(1), body)])
    with pytest.raises((ValueError, RuntimeError, TypeError), match=message):
        run_pass("MaterializeScheduleUnits", value)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("materialized", [False, True])
def test_mixed_kernel_metadata_and_cthread_identity_survive_task_materialization(count, materialized):
    @T.prim_func
    def before(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.MixedKernel(1, sids=count) as (block, sid):
            temp = T.alloc_shared((16,), "float32")
            T.copy(A[sid * 16 : sid * 16 + 16], temp)
            T.copy(temp, B[sid * 16 : sid * 16 + 16])

    value = _bind_target(before)
    if materialized:
        value = _materialize(value)
    result = _check(value)
    after_copies = calls(result, "tl.tileop.ascend_copy")
    before_copies = calls(value, "tl.tileop.ascend_copy")
    assert len(after_copies) == len(before_copies) == 2
    for original, rewritten in zip(before_copies, after_copies):
        ir.assert_structural_equal(original, rewritten)
    assert ir.save_json(value).count("vector_count") == ir.save_json(result).count("vector_count")
    if not materialized:
        left = _materialize(result)
        right = _check(_materialize(value))
        ir.assert_structural_equal(left, right)


@pytest.mark.parametrize("wrapper", [None, "if", "tilelang_assume", "tl.assume_requires_runtime_check"])
def test_empty_stage_retains_its_explicit_stage_and_is_idempotent(wrapper):
    enabled = ir.Var("enabled", "bool")
    body = []
    if wrapper == "if":
        body = [ir.If(enabled, [], None)]
    elif wrapper is not None:
        body = [attribute(enabled, wrapper, integer(1), [])]
    result = _check(_kernel([enabled], [_stage(2, body)]))
    assert _stages(kernel_body(result)) == [2]


@pytest.mark.parametrize("stage", [-1, 1 << 32])
def test_manual_stage_rejects_out_of_range_values_even_when_empty(stage):
    with pytest.raises((ValueError, RuntimeError, TypeError), match="non-negative constant int32"):
        run_pass("MaterializeScheduleUnits", _kernel([], [_stage(stage, [])]))


def test_nested_empty_stages_are_rejected():
    with pytest.raises((ValueError, RuntimeError, TypeError), match="Nested T.Stage"):
        run_pass("MaterializeScheduleUnits", _kernel([], [_stage(2, [_stage(1, [])])]))


def test_multiple_kernel_siblings_keep_same_named_launch_definitions_distinct():
    output = tensor("output", [2], "int32", "global")
    left, right = ir.Var("block", "int32"), ir.Var("block", "int32")
    value = function(
        [output],
        [
            ir.RegionStmt(
                "tirx.launch_thread", [ir.StringImm("blockIdx.x"), integer(1)], [left], {}, [ir.TensorStore(output, [left], integer(1))]
            ),
            ir.RegionStmt(
                "tirx.launch_thread",
                [ir.StringImm("blockIdx.x"), integer(1)],
                [right],
                {},
                [ir.TensorStore(output, [right + 1], integer(2))],
            ),
        ],
    )
    result = _check(value)
    roots = [node for node in _attributes(result, "tl.sblock") if _fields(node)["value"]["name"] == "tilelang_root"]
    assert not roots
    assert len(_attributes(result, "tl.schedule_unit")) == 2
    launches = [node for node in objects(result, ir.RegionStmt) if node.op.name == "tirx.launch_thread"]
    assert launches[0].body_params[0].same_as(left)
    assert launches[1].body_params[0].same_as(right)


def test_partly_materialized_and_malformed_schedule_units_are_rejected():
    variable = ir.Var("value", "int32")
    output = tensor("output", [1], "int32", "global")
    partial = _kernel([output], [ir.Bind(variable, _scheduled_bind(integer(4))), ir.TensorStore(output, [integer(0)], variable)])
    with pytest.raises((ValueError, RuntimeError, TypeError), match="Incomplete native schedule"):
        run_pass("MaterializeScheduleUnits", partial)
    malformed = _kernel(
        [],
        [
            attribute(
                {"stage": integer(-1, "int64")},
                "tl.schedule_unit",
                integer(1),
                [_task([ir.Evaluate(integer(0))]), _task([ir.Evaluate(integer(0))])],
            )
        ],
    )
    with pytest.raises((ValueError, RuntimeError, TypeError), match="exactly one T.Task"):
        run_pass("MaterializeScheduleUnits", malformed)


def test_task_metadata_is_preserved_as_explicit_operands():
    output = tensor("output", [1], "int32", "global")
    offset = ir.Var("offset", "int32")
    metadata = {"latency": integer(5, "int64"), "ii": integer(2, "int64"), "owner_metadata": {output: [offset, region(output)]}}
    value = _kernel([output, offset], [_task([ir.TensorStore(output, [integer(0)], offset)], metadata)])
    result = _check(value)
    (task,) = _attributes(result, "tl.ascend_task")
    ir.assert_structural_equal(_fields(task)["node"], _fields(_attributes(value, "tl.ascend_task")[0])["node"])
    assert any(var.same_as(offset) for var in objects(task.args[0], ir.Var))


def test_unconsumed_scheduled_binding_is_rejected_by_real_ascend_codegen():
    incoming, local = ir.Var("incoming", "int32"), ir.Var("local", "int32")
    value = function([incoming], [ir.Bind(local, _scheduled_bind(incoming + 1))]).with_attr("calling_conv", 2)
    assert tirx.analysis.check_well_formed(value)
    with pytest.raises((ValueError, RuntimeError, TypeError), match="scheduled_bind|Unresolved|unresolved"):
        tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(ir.IRModule({"main": value}), value.attrs["target"])
