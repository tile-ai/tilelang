"""Schedule units contracts."""

from tvm_tirx import ir
from testing.ascend._ir import (
    attribute,
    attribute_fields,
    attribute_key,
    scheduled_fields,
    sblock,
    sblock_annotations,
    sblock_metadata,
    sblocks,
    single_statement,
    tensor,
)

import pytest
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from testing.ascend._ir import copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule._task_utils import (
    _bind_target,
    _collect_schedule_units,
    _collect_task_metadata,
    _make_program,
    _materialize_schedule_units,
)


def _make_guarded_program():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2) as bx:
            temp = T.alloc_shared((16,), "float32")
            if bx == 0:
                T.copy(A, temp)
            else:
                T.copy(temp, B)

    return main


def _make_partially_staged_program():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(4, annotations={"enable_offset": True}):
                # Source order intentionally opposes logical stage order.
                with T.Stage(1), T.Task(latency=1, ii=1):
                    T.copy(temp, B[i * 16 : (i + 1) * 16])
                with T.Task(latency=1, ii=1):
                    T.copy(A[i * 16 : (i + 1) * 16], temp)

    return main


def _direct_schedule_unit_stages(stmt):
    if attribute_key(single_statement(stmt)) == "tl.loop.annotations":
        stmt = single_statement(stmt).body
    statements = stmt.seq if isinstance(stmt, ir.SeqStmt) else [stmt]
    return [
        int(attribute_fields(statement)["node"]["stage"])
        for statement in statements
        if isinstance(statement, ir.RegionStmt) and attribute_key(statement) == "tl.schedule_unit"
    ]


def test_materialize_schedule_units_builds_trivial_guarded_schedule():
    mod = _materialize_schedule_units(_bind_target(_make_guarded_program()))
    schedule_units = _collect_schedule_units(mod)
    task_metadata = _collect_task_metadata(mod)

    assert schedule_units
    statement_units = [unit for unit in schedule_units if isinstance(unit, ir.RegionStmt)]
    binding_units = [unit for unit in schedule_units if isinstance(unit, ir.Bind)]
    assert all(int(attribute_fields(unit)["node"]["stage"]) == -1 for unit in statement_units)
    assert all(len(attribute_fields(unit)["node"]) == 1 for unit in statement_units)
    assert binding_units and all(int(scheduled_fields(unit)["stage"]) == -1 for unit in binding_units)
    guarded_units = [
        node
        for node, parents in statements(mod)
        if attribute_key(node) == "tl.schedule_unit" and any(isinstance(parent, ir.If) for parent in parents)
    ]
    assert len(guarded_units) >= 2
    assert task_metadata["latency"] == []
    assert task_metadata["ii"] == []


def test_materialize_schedule_units_normalizes_partial_manual_stages():
    mod = _materialize_schedule_units(_bind_target(_make_partially_staged_program()))
    loop_units = [
        unit for unit in _collect_schedule_units(mod) if isinstance(unit, ir.RegionStmt) and isinstance(single_statement(unit.body), ir.For)
    ]

    assert len(loop_units) == 1
    assert _direct_schedule_unit_stages(single_statement(loop_units[0].body).body) == [1, 0]


def _scheduled_copies(*, cost=(1, 1), malformed=False, staged=False):
    a = tensor((4, 64), "float32", name="A")
    c = tensor((4, 64), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    producer = unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64]), core=0, cost=cost)
    consumer = unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64]), core=0, stage=1 if staged else 0, cost=cost)
    if malformed:
        body = attribute({"stage": 0}, "tl.schedule_unit", 1, seq(producer.body, consumer.body))
    else:
        body = seq(consumer, producer) if staged else seq(producer, consumer)
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        body,
        annotations={"enable_offset": staged, "num_stages": 2, "multi_buffer_eligible": [ub]},
    )
    return kernel(unit(loop, core=None), buffers=[ub], params=[a, c], allocation_stage=0, allocation_cost=(1, 1))


def test_auto_schedule_preserves_explicit_stage_dependencies():
    before = _scheduled_copies(staged=True)
    assert "tirx.noalias" not in before["main"].attrs
    after = ascend_transform.AutoSchedule()(before)
    (loop_unit,) = [
        node
        for node in _collect_schedule_units(after)
        if isinstance(node, ir.RegionStmt) and isinstance(single_statement(node.body), ir.For)
    ]
    body = single_statement(loop_unit.body).body
    if attribute_key(single_statement(body)) == "tl.loop.annotations":
        body = single_statement(body).body
    units = [node for node in body.seq if attribute_key(node) == "tl.schedule_unit"]
    assert len(units) == 2
    positions = {}
    for phase, node in enumerate(units):
        (operation,) = [call for call in nodes(node, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tl.tileop.ascend_copy"]
        route = (operation.args[0].source.name, operation.args[1].source.name)
        assert route not in positions
        positions[route] = (int(attribute_fields(node)["node"]["stage"]), phase)
    assert set(positions) == {("A", "ub"), ("ub", "C")}
    producer = positions[("A", "ub")]
    consumer = positions[("ub", "C")]
    assert producer[0] == 0 and consumer[0] == 1
    # A and C may overlap: C(k) must complete before the next A tile is read.
    # Requested stages describe logical iterations; source position is the
    # modulo phase, so a sorted list of stage numbers is not this contract.
    for iteration in range(4):
        produce = (iteration + producer[0], producer[1])
        consume = (iteration + consumer[0], consumer[1])
        assert produce < consume
        if iteration + 1 < 4:
            next_produce = (iteration + 1 + producer[0], producer[1])
            assert consume < next_produce


@pytest.mark.parametrize(
    "case, message",
    [
        ("missing-cost", "EstimateLatency"),
        ("missing-units", "MaterializeScheduleUnits"),
        ("multiple-tasks", "exactly one outer T.Task/T.PerCoreTask"),
    ],
)
def test_auto_schedule_requires_its_input_contract(case, message):
    if case == "missing-units":
        before = _bind_target(_make_program())
    else:
        before = _scheduled_copies(cost=None if case == "missing-cost" else (1, 1), malformed=case == "multiple-tasks")
    native_message = {
        "missing-units": "DecodeScheduledTIR found an unmarked Bind",
        "multiple-tasks": "Scheduled leaf must contain exactly one native T.Task/T.PerCoreTask",
    }.get(case, message)
    with pytest.raises((ValueError, RuntimeError), match=native_message):
        ascend_transform.AutoSchedule()(before)


@pytest.mark.parametrize("eligible", [False, True], ids=["ordinary-dependency", "owner-exclusion"])
@pytest.mark.parametrize("second_stage", [0, 1], ids=["same-stage", "offset-stage"])
def test_single_version_sibling_owners_preserve_dependency_kind(eligible, second_stage):
    a = tensor((1,), "int32", name="A")
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    owners = []
    for index in range(2):
        i = ir.Var(f"owner{index}", "int32")
        write = ir.TensorStore(ub, [0], index + 1)
        if index == 0:
            write = sblock("SIMD_VF", write)
        # The shared MTE2 pipe fixes physical owner order. Without it, the
        # ordinary dependency can be satisfied by reversing modulo phases.
        body = seq(unit(copy(a, ub), core=0, stage=-1, cost=(1, 1)), unit(write, core=0, stage=-1, cost=(1, 1)))
        owner = ir.For(
            i,
            ir.const(0),
            ir.const(0) + ir.const(1),
            ir.const(1),
            ir.ForKind.DEFAULT,
            body,
            annotations={"multi_buffer_eligible": [ub]} if eligible else {},
        )
        owners.append(unit(owner, core=None, stage=second_stage if index else 0))
    outer = ir.For(
        ir.Var("outer", "int32"),
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        seq(*owners),
        annotations={"num_stages": 2, "enable_offset": True},
    )
    before = kernel(
        unit(outer, core=None, stage=-1),
        buffers=[ub],
        params=[a],
        annotations={"tl.buffer_versions_map": {ub: 1}},
        allocation_stage=-1,
        allocation_cost=(1, 1),
    )
    if not eligible and second_stage:
        with pytest.raises(RuntimeError, match="Manual schedule constraints are infeasible"):
            ascend_transform.AutoSchedule()(before)
        return
    after = ascend_transform.AutoSchedule()(before)
    (scheduled_outer,) = [loop for loop in nodes(after, ir.For) if loop.var.same_as(outer.var)]
    assert _direct_schedule_unit_stages(scheduled_outer.body) == [0, second_stage]
    root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
    versions = sblock_annotations(root).get("tl.buffer_versions_map", {})
    assert (ub in versions) == eligible
    if eligible:
        assert int(versions[ub]) == 1


@pytest.mark.parametrize(
    "fixed,size", [(False, 96), (True, 96), (False, 100)], ids=["inferred-fits", "fixed-fits", "inferred-exceeds-pool"]
)
def test_ss_versions_share_capacity_across_aivs(fixed, size):
    buffers = [tensor((size,), "uint64", name=f"ss{sid}", scope="shared.ss") for sid in range(2)]
    out = tensor((4, 2), "uint64", name="out")
    i = ir.Var("i", "int32")
    tasks = []
    for sid, (buffer, core) in enumerate(zip(buffers, (1, 4))):
        tasks.append(unit(ir.TensorStore(buffer, [0], ir.prim.Cast("uint64", i)), core=core, cost=(1, 1)))
        tasks.append(unit(ir.TensorStore(out, [i, sid], buffer[0]), core=2, stage=2, cost=(1, 1)))
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        seq(*tasks),
        annotations={"num_stages": 3, "enable_offset": True, "multi_buffer_eligible": [buffer for buffer in buffers]},
    )
    annotations = {"tl.split_subcores": True}
    if fixed:
        annotations["tl.buffer_versions_map"] = {buffer: 2 for buffer in buffers}
    before = kernel(
        unit(loop, core=None), buffers=buffers, params=[out], annotations=annotations, allocation_stage=0, allocation_cost=(1, 1)
    )
    if size == 100:
        with pytest.raises(RuntimeError, match="shared.ss|Manual schedule constraints are infeasible"):
            ascend_transform.AutoSchedule()(before)
        return
    after = ascend_transform.AutoSchedule()(before)
    root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
    versions = sblock_annotations(root)["tl.buffer_versions_map"]
    # Both AIVs consume the same 3 KiB pool. Two versions of each fit exactly.
    assert all(int(versions[buffer]) == 2 for buffer in buffers)


def test_sf_hazard_uses_the_data_ring_variable_in_the_solver():
    # The consumer is two stages later. A one-version SF recurrence cannot
    # satisfy this manual schedule; it must use the group's requested ring.
    data = tensor((32, 128), "float8_e4m3fn", name="data", scope="shared.l0a")
    sf = tensor((32, 2), "uint16", name="sf", scope="shared.l0a.sf")
    source = tensor((32, 128), data.ty.dtype, name="source", scope="shared.l1")
    scale_source = tensor((32, 2), sf.ty.dtype, name="scale_source", scope="shared.l1")
    out = tensor((4,), "uint16", name="out")
    i = ir.Var("i", "int32")
    tasks = seq(
        unit(copy(source, data), core=2, cost=(1, 1)),
        unit(copy(scale_source, sf), core=2, cost=(1, 1)),
        unit(ir.TensorStore(out, [i], sf[0, 0]), core=2, stage=2, cost=(1, 1)),
    )
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        tasks,
        annotations={"num_stages": 3, "enable_offset": True, "multi_buffer_eligible": [data, sf]},
    )
    before = kernel(
        unit(loop, core=None),
        buffers=[data, sf, source, scale_source],
        params=[out],
        annotations={
            "tl.l0_sf_bindings": {sf: data},
            "tl.buffer_versions_map": {sf: 3},
        },
        allocation_stage=0,
        allocation_cost=(1, 1),
    )
    after = ascend_transform.AutoSchedule()(before)
    root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
    versions = sblock_annotations(root)["tl.buffer_versions_map"]
    assert int(versions[data]) == int(versions[sf]) == 3


@pytest.mark.parametrize("owners", [1, 2], ids=["loop-local", "sibling-identities"])
def test_inferred_versions_remain_inside_the_storage_definition_scope(owners):
    import tvm_ffi
    from tvm_tirx import tirx
    from testing.ascend._ir import allocate, annotation, annotations

    a = tensor((owners * 4, 64), "float32", name="A")
    c = tensor((owners * 4, 64), "float32", name="C")
    loops, storages = [], []
    for owner in range(owners):
        i = ir.Var("i", "int32")
        ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
        storages.append(ub)
        body = [
            unit(allocate(ub), core=7, cost=(1, 1)),
            unit(copy(a, ub, src_indices=[owner * 4 + i, 0], src_shape=[1, 64]), core=5, cost=(1, 1)),
            unit(copy(ub, c, dst_indices=[owner * 4 + i, 0], dst_shape=[1, 64]), core=5, cost=(1, 1)),
            annotation({"multi_buffer_eligible": [ub]}),
        ]
        loops.append(unit(ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, body), core=None))
    before = kernel(seq(*loops), params=[a, c])
    assert tirx.analysis.check_well_formed(before)
    serialized = ir.save_json(before)
    after = ascend_transform.AutoSchedule()(before)
    assert ir.save_json(before) == serialized
    assert tirx.analysis.check_well_formed(after)

    def check_versions(module):
        entries = [entry for fields in annotations(module) for entry in fields.get("tl.buffer_versions_map", {}).items()]
        versions = dict(entries)
        assert len(entries) == len(versions) == owners
        for storage in storages:
            assert int(versions[storage]) >= 1
            carrier = next(
                statement
                for statement in nodes(module, ir.Evaluate)
                if isinstance(statement.value, ir.Call)
                and statement.value.op.name == "tl.annotation"
                and storage
                in tvm_ffi.get_global_func("tl.DecodeAnnotations")(statement.value.args[0], statement.value.attrs.annotation_schema).get(
                    "tl.buffer_versions_map", {}
                )
            )
            parents = next(parents for item, parents in statements(module) if item.same_as(carrier))
            assert any(isinstance(parent, ir.For) for parent in parents)
        return versions

    expected = check_versions(after)
    # Later scheduled passes must read the nested carriers back into the same
    # analysis map, without losing entries or moving them outside the loop.
    resolved = ascend_transform.ResolveCore()(after)
    assert tirx.analysis.check_well_formed(resolved)
    assert check_versions(resolved) == expected
