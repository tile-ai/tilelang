"""The real loop solver budgets complete native physical allocations."""

import importlib
import json
import os
from pathlib import Path

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from test_native_ascend_core_passes import _apply, _core, _costed, _metadata, _run
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_ops import function, strided_tensor
from test_native_ascend_schedule_units import _bind_target
from test_native_reducer_contracts import allocate, annotation, integer, objects, region, tensor, tileop


CAPACITY = 248 * 1024


def _solver():
    return importlib.import_module("tilelang.ascend.transform.z3_scheduler")


def _view(root, name, dtype, shape, byte_offset):
    value = tensor(name, shape, dtype, "shared")
    pointer = ir.Call("tirx.ptr_byte_offset", [root.data, integer(byte_offset, "int64")], ty=root.data.ty)
    return value, ir.Bind(value, ir.Call("tirx.decl_tensor", [pointer], ty_args=[value.ty]))


def _case(storage, *, aliases=(), access=None, stages=2, manual=False, budget=None, fixed=None, params=()):
    _solver()
    access = storage if access is None else access
    dtype = str(access.ty.dtype)
    output = tensor("output", [4], dtype, "local" if manual else "global")
    index = ir.Var("index", "int32")
    definitions = [_costed(allocate(storage))]
    if manual:
        # A fresh output allocation proves disjointness for the staged
        # producer/consumer pair. External parameters may alias on-chip roots.
        definitions.append(_costed(allocate(output)))
    definitions.extend(_costed(binding) for _, binding in aliases)
    body = []
    if budget is not None:
        reserve = tensor("reserve", [CAPACITY - budget], "uint8", "shared")
        definitions.append(_costed(allocate(reserve)))
        body.append(_costed(ir.TensorStore(reserve, [integer(0)], ir.const(0, "uint8")), stage=0 if manual else -1))
    body.extend(
        [
            _costed(ir.TensorStore(access, [integer(0)], ir.const(1, dtype)), stage=0 if manual else -1),
            _costed(ir.TensorStore(output, [index], ir.TensorLoad(access, [integer(0)])), stage=stages - 1 if manual else -1),
        ]
    )
    loop = ir.For(
        index,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        body,
        {
            "multi_buffer_eligible": [storage],
            "num_stages": integer(stages) if isinstance(stages, int) else stages,
            "enable_offset": True,
        },
    )
    value = _kernel(
        [*params] if manual else [*params, output], [*definitions, _unit(loop, stage=-1)], {storage: fixed} if fixed else None, count=1
    )
    return _bind_target(value)


def _selected(value, name):
    found = []
    for fields in _metadata(value):
        for storage, count in fields.get("tl.buffer_versions_map", {}).items():
            if storage.name == name:
                found.append(int(count))
    assert found
    assert len(set(found)) == 1
    return found[0]


def _record(name, value, facts):
    destination = Path(os.environ["TILELANG_OPS_ARTIFACT_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}-input.json").write_text(ir.save_json(value))
    (destination / f"{name}-oracle.json").write_text(json.dumps(facts, indent=2) + "\n")


@pytest.mark.parametrize("kind", ["padded", "small_alias"])
def test_real_z3_rejects_root_that_exceeds_capacity(kind):
    if kind == "padded":
        storage = strided_tensor("large_physical", [2], [65536], 0, "float32", "shared")
        value = _case(storage)
        physical = 65537 * 4
    else:
        storage = tensor("large_physical", [65536], "float32", "shared")
        alias, binding = _view(storage, "small_alias", "float32", [2], 4)
        value = _case(storage, aliases=[(alias, binding)], access=alias)
        physical = 65536 * 4
    _record("capacity-" + kind, value, {"physical_bytes": physical, "capacity": CAPACITY})
    assert physical > CAPACITY
    with pytest.raises(RuntimeError, match="capacity|memory|insufficient"):
        _run("AutoSchedule", value)


def test_one_version_can_exactly_fill_the_physical_capacity():
    storage = strided_tensor("boundary", [2], [CAPACITY // 4 - 1], 0, "float32", "shared")
    value = _case(storage)
    scheduled = _run("AutoSchedule", value)
    assert _selected(scheduled, storage.name) == 1


@pytest.mark.parametrize("fixed", [False, True])
def test_two_versions_count_ub_padding_and_origin_before_scheduling(fixed):
    storage = strided_tensor("padded_origin", [2], [126972], 2, "uint8", "shared")
    value = _case(storage, manual=not fixed, fixed=2 if fixed else None)
    _record(
        "capacity-padded-origin-" + str(fixed),
        value,
        {
            "one_version_bytes": 126975,
            "unrounded_two_versions": 253950,
            "physical_two_versions": 2 + 126976 * 2,
            "capacity": CAPACITY,
        },
    )
    assert 126975 * 2 <= CAPACITY < 2 + 126976 * 2
    with pytest.raises(RuntimeError, match="capacity|memory|insufficient|infeasible"):
        _run("AutoSchedule", value)


def _footprint_case(kind, versions):
    aliases = []
    access = None
    if kind == "compact":
        storage = tensor("storage", [3], "uint8", "shared")
        expected = 3 if versions == 1 else 32 * versions
    elif kind == "origin":
        storage = strided_tensor("storage", [2], [5], 3, "uint8", "shared")
        expected = 9 if versions == 1 else 3 + 32 * versions
    elif kind == "packed":
        storage = tensor("storage", [3], "float4_e2m1fn", "shared")
        expected = 2 if versions == 1 else 32 * versions
    elif kind == "alias_alignment":
        storage = tensor("storage", [65], "uint8", "shared")
        alias, binding = _view(storage, "wide_alias", "float32x16", [1], 1)
        aliases = [(alias, binding)]
        expected = 65 + (versions - 1) * 128
    elif kind == "small_alias":
        storage = tensor("storage", [32], "float32", "shared")
        alias, binding = _view(storage, "small_alias", "uint8", [2], 4)
        aliases = [(alias, binding)]
        access = alias
        expected = 128 * versions
    elif kind == "bare_swizzle":
        layout = tirx.ComposeLayout(0, 1, 2, tirx.TileLayout.from_iters(), swizzle_inner=False)
        storage = ir.Var("storage", tirx.TensorType([17], "int32", scope="shared", layout=layout))
        # A bare swizzle describes allocation storage. Native logical accesses
        # use an explicit view rather than applying an empty tile map.
        access, binding = _view(storage, "bare_alias", "int32", [17], 0)
        aliases = [(access, binding)]
        expected = 96 * versions
    elif kind == "replica":
        layout = tirx.TileLayout(tirx.layout.S[(2,) : (3,)] + tirx.layout.R[(2,) : (8,)] + 1)
        storage = ir.Var("storage", tirx.TensorType([2], "uint8", scope="shared", layout=layout))
        expected = 13 if versions == 1 else 1 + 32 * versions
    else:
        assert kind == "swizzle"
        tile = tirx.TileLayout(tirx.layout.S[(6,) : (3,)] + 1)
        layout = tirx.ComposeLayout(0, 1, 2, tile, swizzle_inner=False)
        storage = ir.Var("storage", tirx.TensorType([6], "int32", scope="shared", layout=layout))
        expected = 96 * versions
    return storage, aliases, access, expected


@pytest.mark.parametrize("kind", ["compact", "origin", "packed", "alias_alignment", "small_alias", "swizzle", "bare_swizzle", "replica"])
@pytest.mark.parametrize("versions", [1, 2, 3])
def test_each_version_cost_matches_physical_consumers_at_the_budget_boundary(kind, versions):
    storage, aliases, access, expected = _footprint_case(kind, versions)
    value = _case(storage, aliases=aliases, access=access, stages=versions, manual=True, budget=expected)
    scheduled = _run("AutoSchedule", value)
    assert _selected(scheduled, storage.name) == versions

    # A second real allocation reserves the rest of UB. Reducing the remaining
    # capacity by one byte must make this exact staged schedule infeasible.
    insufficient = _case(storage, aliases=aliases, access=access, stages=versions, manual=True, budget=expected - 1)
    with pytest.raises(RuntimeError, match="capacity|memory|insufficient|infeasible"):
        _run("AutoSchedule", insufficient)

    assert _physical_bytes(scheduled, storage.name) == expected


def _physical_bytes(scheduled, storage_name):
    # Compare the accepted budget against real version and native layout
    # consumers, before arena merging obscures each allocation's own size.
    for name in ("PrepareMultiBuffer", "ResolveCore", "InsertSync", "MaterializeMultiBuffer", "LowerScheduledTIR"):
        scheduled = _run(name, scheduled)
    scheduled = _bind_target(function(list(scheduled.params), _core(scheduled, "VECTOR").body))
    for name in ("NormalizeBufferVersion", "RewriteAscendBufferVersionLayout"):
        scheduled = _run(name, scheduled)
    scheduled = _apply(tirx.transform.LowerTensorLayouts(), scheduled)
    allocations = [
        bind.var
        for bind in objects(scheduled, ir.Bind)
        if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor" and bind.var.name == storage_name
    ]
    assert len(allocations) == 1
    physical = allocations[0]
    elements = int(physical.ty.shape[0])
    dtype = tvm_ffi.dtype(str(physical.ty.dtype))
    return (elements * dtype.bits * dtype.lanes + 7) // 8 if dtype.bits == 4 else elements * ((dtype.bits + 7) // 8) * dtype.lanes


@pytest.mark.parametrize("versions", [1, 2, 3])
def test_empty_native_allocation_stays_empty_with_a_full_ub_reservation(versions):
    _solver()
    storage = tensor("empty", [0], "uint8", "shared")
    reserve = tensor("reserve", [CAPACITY], "uint8", "shared")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        [
            _costed(tileop("fill", region(storage), ir.const(0, "uint8"))),
            _costed(ir.TensorStore(reserve, [integer(0)], ir.const(0, "uint8"))),
        ],
        {"multi_buffer_eligible": [storage], "num_stages": integer(versions)},
    )
    value = _bind_target(
        _kernel([], [_costed(allocate(storage)), _costed(allocate(reserve)), _unit(loop, stage=-1)], {storage: versions}, count=1)
    )
    scheduled = _run("AutoSchedule", value)
    assert _selected(scheduled, storage.name) == versions
    assert _physical_bytes(scheduled, storage.name) == 0


@pytest.mark.parametrize("kind", ["dynamic", "element_overflow", "span_overflow"])
def test_unprovable_or_overflowing_native_footprints_are_rejected(kind):
    params = []
    if kind == "dynamic":
        extent = ir.Var("extent", "int64")
        params = [extent]
        storage = tensor("unknown", [extent], "uint8", "shared")
    elif kind == "element_overflow":
        storage = tensor("huge", [integer(1 << 62, "int64")], "float32", "shared")
    else:
        storage = strided_tensor("huge_span", [integer(2, "int64")], [integer((1 << 63) - 1, "int64")], 1, "uint8", "shared")
    with pytest.raises(ValueError, match="static physical memory footprint|overflows int64"):
        _run("AutoSchedule", _case(storage, params=params))


@pytest.mark.parametrize("stages", [0, -1, (1 << 32) + 1, "dynamic"])
def test_native_stage_count_is_checked_before_narrowing(stages):
    if stages == "dynamic":
        count = ir.Var("stage_count", "int64")
        params = [count]
    else:
        count = integer(stages, "int64")
        params = []
    with pytest.raises(ValueError, match="num_stages"):
        _run("AutoSchedule", _case(tensor("storage", [2], "uint8", "shared"), stages=count, params=params))


@pytest.mark.parametrize("alias", [False, True])
def test_existing_manual_versions_include_padding_without_adding_another_dimension(alias):
    _solver()
    root = tensor("manual", [6] if alias else [2, 3], "uint8", "shared")
    logical = root
    declarations = [_costed(allocate(root))]
    if alias:
        logical, binding = _view(root, "ring", "uint8", [2, 3], 0)
        declarations.append(_costed(binding))
    automatic = tensor("automatic", [1], "uint8", "shared")
    output = tensor("output", [4], "uint8", "local")
    index = ir.Var("index", "int32")
    for budget in (65, 64):
        reserve = tensor("reserve", [CAPACITY - budget], "uint8", "shared")
        loop = ir.For(
            index,
            integer(0),
            integer(4),
            integer(1),
            ir.ForKind.DEFAULT,
            [
                _costed(ir.TensorStore(logical, [integer(0), integer(0)], ir.const(1, "uint8"))),
                _costed(ir.TensorStore(automatic, [integer(0)], ir.TensorLoad(logical, [integer(0), integer(0)]))),
                _costed(ir.TensorStore(output, [index], ir.TensorLoad(automatic, [integer(0)]))),
                _costed(ir.TensorStore(reserve, [integer(0)], ir.const(0, "uint8"))),
            ],
            {"multi_buffer_eligible": [automatic], "num_stages": integer(1)},
        )
        value = _bind_target(
            _kernel(
                [],
                [
                    *declarations,
                    _costed(allocate(automatic)),
                    _costed(allocate(output)),
                    _costed(allocate(reserve)),
                    annotation({"tl.manual_multi_buffer": {root: 2}}),
                    _unit(loop, stage=-1),
                ],
                count=1,
            )
        )
        if budget == 64:
            with pytest.raises(RuntimeError, match="capacity|memory|insufficient"):
                _run("AutoSchedule", value)
        else:
            scheduled = _run("AutoSchedule", value)
            assert _selected(scheduled, automatic.name) == 1
            assert _physical_bytes(scheduled, root.name) == 64


def _direct_solver(*, costs=None, capacity=64, stages=2, size=2, force_versions=True, ffi=False):
    arguments = (
        stages,
        [1, 1],
        [1, 1],
        [1, 1],
        [(1, 0, -1, 1)] if force_versions else [],
        [(0, 1)],
        [],
        [(0, 1)],
        [size],
        [[capacity, 0]],
    )
    manual_stages = [0, stages - 1] if force_versions else [0, 0]
    if ffi:
        # Registered packed functions accept positional arguments. Omitting
        # the appended cost table exercises the original FFI call shape.
        return _solver().z3_schedule_loop_ffi(*arguments, [], True, manual_stages, True, *([] if costs is None else [costs]))
    return _solver().z3_schedule_loop_python(
        *arguments,
        enable_offset=True,
        manual_stages=manual_stages,
        manual_schedule=True,
        buffer_version_bytes=costs,
    )


@pytest.mark.parametrize("ffi", [False, True])
def test_scalar_solver_contract_remains_available(ffi):
    _, versions, _ = _direct_solver(capacity=4, ffi=ffi)
    assert versions == [2]


def test_non_linear_cost_table_is_used_by_the_real_z3_constraints():
    _, versions, _ = _direct_solver(costs=[[2, 64]])
    assert versions == [2]
    with pytest.raises(RuntimeError, match="infeasible|capacity"):
        _direct_solver(costs=[[2, 64]], capacity=63)


def test_empty_allocation_cost_is_zero_for_every_version():
    _, versions, _ = _direct_solver(costs=[[0, 0]], size=0, capacity=0)
    assert versions == [2]


@pytest.mark.parametrize(
    "costs",
    [
        [],
        [[2]],
        [[2, 64, 96]],
        [[2, -1]],
        [[2, 1.5]],
        [[2, True]],
        [[2, 1 << 63]],
        [[3, 64]],
        [[2, 1]],
        [None],
        [2],
        [{0: 2, 1: 64}],
        "bad",
        2,
    ],
)
def test_invalid_physical_cost_tables_are_rejected_before_solver_search(costs):
    with pytest.raises(ValueError, match="buffer_version_bytes"):
        _direct_solver(costs=costs)


@pytest.mark.parametrize("stages", [0, -1, True])
def test_invalid_stage_count_is_rejected_before_solver_search(stages):
    with pytest.raises(ValueError, match="num_stages"):
        _direct_solver(stages=stages)
