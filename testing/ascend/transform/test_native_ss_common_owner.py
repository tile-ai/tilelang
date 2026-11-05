"""SS storage has one physical owner across independently renewed core streams."""

import pytest
from testing.ascend._ir import (
    allocate,
    allocated_tensors,
    annotation,
    annotations,
    kernel,
    launch_thread,
    nodes,
    sblocks,
    substitute,
    tensor,
)
from testing.ascend._memory_ir import address_bits, storage_owner
from testing.ascend.auto_schedule._scheduled_ir import unit
from tilelang.ascend import transform

import tvm_tirx as tvm
from tvm_tirx import ir, tirx


def lower(before):
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    after = transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    assert not tirx.analysis.undefined_vars(after["main"])
    return after


def ss_allocations(value):
    return [var for var in allocated_tensors(value) if var.ty.scope.name in ("shared.ss", "shared.ss.dyn")]


def assert_common_owner(root, expected):
    allocations = ss_allocations(root)
    assert len(allocations) == len(expected)
    assert all(actual.same_as(original) for actual, original in zip(allocations, expected))
    prefix = list(root.body.seq[: len(expected)])
    assert all(isinstance(stmt, ir.Bind) and stmt.var.same_as(var) for stmt, var in zip(prefix, expected))
    for name in ("CUBE", "VECTOR"):
        for core in sblocks(root, name):
            assert not ss_allocations(core)


@pytest.mark.parametrize("scope", ["shared.ss", "shared.ss.dyn"])
def test_common_owner_survives_typed_views_and_physical_version_metadata(scope):
    words = tensor((3, 2), "uint32", name="words", scope=scope, align=32)
    packed = tensor((3, 1), "uint64", name="packed", scope=scope, align=1)
    output = tensor((2,), "uint64", name="output")
    body = [
        unit(ir.Bind(packed, ir.Call("tirx.decl_tensor", [words.data], ty_args=[packed.ty])), core=7),
        annotation({"tl.buffer_version": {words: 3, packed: 3}, "pragma_owner": 19}),
        unit(ir.TensorStore(words, [1, 0], ir.const(73, "uint32")), core=1),
        unit(ir.TensorStore(output, [0], ir.TensorLoad(packed, [1, 0])), core=2),
        unit(ir.TensorStore(output, [1], ir.TensorLoad(packed, [1, 0])), core=1),
    ]
    after = lower(kernel(body, buffers=[words], params=[output], allocation_stage=0))
    (root,) = sblocks(after, "tilelang_root")
    assert_common_owner(root, [words])
    aliases = []
    for name in ("CUBE", "VECTOR"):
        (core,) = sblocks(root, name)
        (alias,) = [binding.var for binding in nodes(core, ir.Bind) if binding.var.name == "packed"]
        aliases.append(alias)
        assert not alias.same_as(packed)
        (fields,) = [fields for fields in annotations(core) if "tl.buffer_version" in fields]
        versions = fields["tl.buffer_version"]
        assert len(versions) == 2 and int(versions[words]) == 3 and int(versions[alias]) == 3
        assert int(fields["pragma_owner"]) == 19
        (load,) = nodes(core, ir.TensorLoad)
        owner, offset = address_bits(after, load)
        assert owner.same_as(words) and int(offset) == 64
    assert not aliases[0].same_as(aliases[1])
    (store,) = [store for store in nodes(after, ir.TensorStore) if store.dest.ty.scope.name == scope]
    assert storage_owner(after, store).same_as(words)


@pytest.mark.parametrize("core", [1, 2, 4, 5])
def test_single_core_uses_one_common_ss_definition(core):
    words = tensor((2,), "uint32", name="words", scope="shared.ss", align=64)
    after = lower(
        kernel(
            [unit(ir.TensorStore(words, [0], ir.const(9, "uint32")), core=core)],
            buffers=[words],
            allocation_stage=0,
            allocation_core=core,
        )
    )
    (root,) = sblocks(after, "tilelang_root")
    assert_common_owner(root, [words])
    assert len(sblocks(root, "CUBE")) == (core == 2)
    assert len(sblocks(root, "VECTOR")) == (core != 2)
    assert int(words.ty.data_alignment) == 64
    assert all(store.dest.same_as(words) for store in nodes(root, ir.TensorStore))


def test_distinct_same_named_ss_allocations_are_not_coalesced():
    owners = [tensor((2,), "uint32", name="words", scope="shared.ss") for _ in range(2)]
    output = tensor((2,), "uint32", name="output")
    body = [
        unit(ir.TensorStore(output, [index], ir.TensorLoad(owner, [0])), core=core) for core in (2, 1) for index, owner in enumerate(owners)
    ]
    after = lower(kernel(body, buffers=owners, params=[output], allocation_stage=0))
    (root,) = sblocks(after, "tilelang_root")
    assert_common_owner(root, owners)
    assert not owners[0].same_as(owners[1])
    for name in ("CUBE", "VECTOR"):
        (core,) = sblocks(root, name)
        for index, load in enumerate(nodes(core, ir.TensorLoad)):
            assert storage_owner(after, load).same_as(owners[index])


@pytest.mark.parametrize("outer_sid", [False, True])
def test_sibling_kernels_keep_distinct_group_owners_after_sid_relocation(outer_sid):
    output = tensor((2,), "uint32", name="output")
    owners = [tensor((2,), "uint32", name="words", scope="shared.ss") for _ in range(2)]
    modules = []
    for owner in owners:
        modules.append(
            kernel(
                [unit(ir.TensorStore(output, [i], ir.TensorLoad(owner, [0])), core=core) for i, core in enumerate((2, 1))],
                buffers=[owner],
                params=[output],
                allocation_stage=0,
            )
        )
    roots = [sblocks(module, "tilelang_root")[0] for module in modules]
    body = [launch_thread(ir.Var("sid", "int32"), "cthread", 2, roots)] if outer_sid else roots
    before = ir.IRModule({"main": modules[0]["main"].with_body(body)})
    after = lower(before)
    assert len(ss_allocations(after)) == 2
    for root, owner in zip(sblocks(after, "tilelang_root"), owners):
        assert_common_owner(root, [owner])
        assert all(storage_owner(after, load).same_as(owner) for load in nodes(root, ir.TensorLoad))


def scalar(expression, environment):
    replacements = {var: ir.IntImm(var.ty, value) for var, value in environment.items()}
    result = tvm.sym.Analyzer().simplify(substitute(expression, replacements))
    assert isinstance(result, ir.IntImm), result
    return int(result)


def stores_in_execution_order(statement, environment):
    """Evaluate finite native guards/loops, retaining each actual storage Var."""
    if isinstance(statement, ir.SeqStmt):
        return [event for child in statement.seq for event in stores_in_execution_order(child, environment)]
    if isinstance(statement, ir.RegionStmt):
        if statement.op.name == "tirx.launch_thread":
            (var,) = statement.body_params
            return [
                event
                for sid in range(scalar(statement.args[1], environment))
                for event in stores_in_execution_order(statement.body, {**environment, var: sid})
            ]
        return stores_in_execution_order(statement.body, environment)
    if isinstance(statement, ir.For):
        return [
            event
            for index in range(*(scalar(value, environment) for value in (statement.begin, statement.end, statement.step)))
            for event in stores_in_execution_order(statement.body, {**environment, statement.var: index})
        ]
    if isinstance(statement, ir.If):
        selected = statement.then_case if scalar(statement.condition, environment) else statement.else_case
        return [] if selected is None else stores_in_execution_order(selected, environment)
    if isinstance(statement, ir.TensorStore):
        return [
            (
                statement.dest,
                tuple(scalar(index, environment) for index in statement.indices),
                scalar(statement.value, environment),
            )
        ]
    assert isinstance(statement, (ir.Bind, ir.Evaluate)), type(statement)
    return []


@pytest.mark.parametrize("placement", ["root", "branch", "loop"])
@pytest.mark.parametrize("rounds,stage", [(0, 2), (4, 0), (4, 2)])
def test_ss_hoisting_preserves_conditional_and_pipeline_accesses(placement, rounds, stage):
    words = tensor((2,), "uint32", name="words", scope="shared.ss")
    enabled = ir.Var("enabled", "bool")
    index = ir.Var("i", "int32")
    tasks = [
        unit(
            ir.TensorStore(words, [0], ir.prim.Cast(ir.PrimType("uint32"), index + 1)),
            core=1,
            guard=index % 2 == 0,
        ),
        unit(
            ir.TensorStore(words, [1], ir.prim.Cast(ir.PrimType("uint32"), index + 11)),
            core=2,
            stage=stage,
            guard=index % 2 == 0,
        ),
    ]
    definition = unit(allocate(words), core=7)
    if placement == "loop":
        tasks.insert(0, definition)
    loop = unit(ir.For(index, 0, rounds, 1, ir.ForKind.DEFAULT, tasks), core=None)
    conditional = ir.If(enabled, [definition, loop] if placement == "branch" else [loop], None)
    body = [definition, conditional] if placement == "root" else [conditional]
    after = lower(kernel(body, params=[enabled]))
    (root,) = sblocks(after, "tilelang_root")
    assert_common_owner(root, [words])
    for enabled_value in (0, 1):
        for name, slot, increment in (("VECTOR", 0, 1), ("CUBE", 1, 11)):
            (core,) = sblocks(root, name)
            events = stores_in_execution_order(core, {enabled: enabled_value})
            expected = [(slot, i + increment) for i in range(rounds) if enabled_value and i % 2 == 0]
            assert [(indices[0], value) for _, indices, value in events] == expected
            assert all(owner.same_as(words) for owner, _, _ in events)


@pytest.mark.parametrize("scope", ["local", "shared.dyn", "shared.l1"])
def test_non_ss_storage_remains_private_to_each_renewed_core(scope):
    words = tensor((2,), "uint32", name="words", scope="shared.ss")
    private = tensor((2,), "uint32", name="private", scope=scope)
    output = tensor((2,), "uint32", name="output")
    body = [
        unit(
            ir.TensorStore(output, [index], ir.TensorLoad(words, [0]) + ir.TensorLoad(private, [0])),
            core=core,
        )
        for index, core in enumerate((2, 1))
    ]
    after = lower(kernel(body, buffers=[words, private], params=[output], allocation_stage=0))
    (root,) = sblocks(after, "tilelang_root")
    assert_common_owner(root, [words])
    private_owners = []
    for name in ("CUBE", "VECTOR"):
        (core,) = sblocks(root, name)
        (owner,) = allocated_tensors(core)
        private_owners.append(owner)
        assert owner.ty.scope.name == scope and not owner.same_as(private)
        (load,) = [load for load in nodes(core, ir.TensorLoad) if load.source.ty.scope.name == scope]
        assert load.source.same_as(owner)
    assert not private_owners[0].same_as(private_owners[1])


@pytest.mark.parametrize("type_dependency", ["parameter-shape", "loop-shape", "mutable-layout"])
def test_non_self_contained_ss_allocation_cannot_be_hoisted(type_dependency):
    extent = ir.Var("extent", "int32")
    metadata = tensor((1,), "int32", name="metadata", scope="local")
    layout = (
        tirx.TileLayout(tirx.layout.S[(2,) : (1,)] + ir.TensorLoad(metadata, [0])) if type_dependency == "mutable-layout" else "default"
    )
    words = tensor(
        (2,) if type_dependency == "mutable-layout" else (extent,),
        "uint32",
        name="words",
        scope="shared.ss",
        layout=layout,
    )
    body = [
        unit(allocate(words), core=7),
        unit(ir.TensorStore(words, [0], ir.const(1, "uint32")), core=1),
    ]
    params = []
    if type_dependency == "parameter-shape":
        params.append(extent)
    elif type_dependency == "loop-shape":
        body = [unit(ir.For(extent, 1, 4, 1, ir.ForKind.DEFAULT, body), core=None)]
    else:
        body.insert(0, unit(allocate(metadata), core=7))
    before = kernel(body, params=params)
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    with pytest.raises(ValueError, match="Group-shared SS allocations require static, self-contained native types"):
        transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
