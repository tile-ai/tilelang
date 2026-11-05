"""Native dependency proofs agree with physical addresses and lexical execution."""

import itertools

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_ops import function, strided_tensor, subregion
from test_native_ascend_schedule_units import _attributes
from test_native_reducer_contracts import allocate, annotation, attribute, integer, objects, region, tensor


def _api(name):
    return tvm_ffi.get_global_func("tl.testing." + name)


def _root(value):
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    ir.assert_structural_equal(value, ir.load_json(ir.save_json(value)))
    return _attributes(value, "tl.sblock")[0]


def _inspect(value, manual=None, owners=None):
    root = _root(value)
    before = ir.save_json(root)
    result = _api("AnalyzeAscendDependencies")(root, manual or {}, owners or {})
    assert ir.save_json(root) == before
    return result


def _ids(result, anchor, *, header=None):
    return [
        int(row["id"])
        for row in result["tasks"]
        if (header is None or bool(row["header"]) == header)
        and any(item.same_as(anchor) for item in objects(row["statement"], type(anchor)))
    ]


def _pairs(result, *, clock=None, root_only=False):
    rows = []
    scopes = result["scopes"][:1] if root_only else result["scopes"]
    for scope in scopes:
        if clock is not None and ("clock" not in scope or not scope["clock"].same_as(clock)):
            continue
        for dependency in scope["dependencies"]:
            for pair in dependency["pairs"]:
                rows.append((int(pair[0]), int(pair[1]), int(dependency["distance"]), dependency))
    return rows


def _declare(view, base, byte_offset=0):
    pointer = ir.Call("tirx.ptr_byte_offset", [base.data, integer(byte_offset, "int64")], ty=base.data.ty)
    return ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))


def _conflict(lhs, rhs, definitions=(), left_conditions=(), right_conditions=(), pitches=(None, None)):
    body = ir.SeqStmt([*definitions, ir.Evaluate(ir.Tuple([lhs, rhs]))])
    assert tirx.analysis.check_well_formed(function(tirx.analysis.undefined_vars(body), [body]))
    before = ir.save_json(body)
    result = _api("AscendRegionsMayConflict")(ir.SeqStmt(definitions), lhs, rhs, list(left_conditions), list(right_conditions), *pitches)
    assert ir.save_json(body) == before
    return result


@pytest.mark.parametrize("dtype,bits", [("int32", 32), ("float16", 16), ("float4_e2m1fn", 4)])
@pytest.mark.parametrize("pattern", ["disjoint", "overlap", "interleaved", "reverse", "broadcast", "pointer", "rank"])
def test_physical_region_overlap_matches_enumerated_bits(dtype, bits, pattern):
    base = tensor("allocation", [128], dtype, "global")
    left = ([4], [1], 0, 0, [0], [2])
    right = ([4], [1], 0, 0, [2], [2])
    if pattern == "overlap":
        right = ([4], [1], 0, 0, [1], [2])
    elif pattern == "interleaved":
        left, right = ([4], [2], 0, 0, [0], [4]), ([4], [2], 1, 0, [0], [4])
    elif pattern == "reverse":
        left = ([4], [-1], 3, 0, [0], [2])
    elif pattern == "broadcast":
        left, right = ([4], [0], 3, 0, [0], [1]), ([4], [0], 3, 0, [3], [1])
    elif pattern == "pointer":
        right = ([4], [1], 0, 1, [0], [2])
    elif pattern == "rank":
        left, right = ([2, 2], [3, 1], 1, 0, [0, 0], [2, 2]), ([4], [1], 2, 0, [0], [4])

    def make(name, spec):
        shape, strides, origin, byte_offset, starts, extents = spec
        view = strided_tensor(name, shape, strides, origin, dtype, "global")
        addresses = set()
        for point in itertools.product(*(range(start, start + extent) for start, extent in zip(starts, extents))):
            start = byte_offset * 8 + (origin + sum(index * stride for index, stride in zip(point, strides))) * bits
            addresses.update(range(start, start + bits))
        return view, _declare(view, base, byte_offset), subregion(view, starts, extents), addresses

    _, left_definition, left_region, left_bits = make("left", left)
    _, right_definition, right_region, right_bits = make("right", right)
    assert _conflict(left_region, right_region, [allocate(base), left_definition, right_definition]) == bool(left_bits & right_bits)


@pytest.mark.parametrize("byte_index,overlap", [(0, True), (1, False)])
def test_fp4_alias_and_byte_alias_compare_actual_element_widths(byte_index, overlap):
    base = tensor("base", [16], "uint8", "global")
    nibble = tensor("nibble", [32], "float4_e2m1fn", "global")
    byte = tensor("byte", [16], "uint8", "global")
    assert (
        _conflict(
            subregion(nibble, [1], [1]),
            subregion(byte, [byte_index], [1]),
            [allocate(base), _declare(nibble, base), _declare(byte, base)],
        )
        == overlap
    )


@pytest.mark.parametrize("origin", ["tensor_parameter", "pointer_parameter", "integer", "fresh", "fresh_and_integer"])
def test_different_roots_need_fresh_allocation_provenance(origin):
    left, right = tensor("left", [4], "int32", "global"), tensor("right", [4], "int32", "global")
    definitions, params = [], []
    for index, view in enumerate([left, right]):
        if origin == "fresh" or (origin == "fresh_and_integer" and index == 0):
            definitions.append(allocate(view))
        elif origin == "tensor_parameter":
            params.append(view)
        else:
            pointer_type = ir.PtrType("int32", "global")
            pointer = ir.Var("root", pointer_type if origin == "pointer_parameter" else "uint64")
            params.append(pointer)
            address = pointer if origin == "pointer_parameter" else ir.Call("tirx.reinterpret", [pointer], ty=pointer_type)
            definitions.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [address], ty_args=[view.ty])))
    assert _conflict(region(left), region(right), definitions) == (origin != "fresh")
    writer, reader = ir.TensorStore(left, [integer(0)], integer(1)), ir.TensorLoad(right, [integer(0)])
    value = _kernel(params, [*map(_unit, definitions), _unit(writer), _unit(ir.Evaluate(reader))])
    result = _inspect(value)
    source, destination = _ids(result, writer)[0], _ids(result, reader)[0]
    matching = [row for row in _pairs(result, root_only=True) if row[:2] == (source, destination)]
    assert bool(matching) == (origin != "fresh")
    assert all("storage" not in row[3] for row in matching)


@pytest.mark.parametrize("accesses,expected", [("wr", True), ("rw", True), ("ww", True), ("rr", False)])
def test_dependency_directions_preserve_raw_war_and_waw(accesses, expected):
    storage = tensor("storage", [4], "int32", "global")
    body = [
        ir.TensorStore(storage, [integer(0)], integer(1)) if access == "w" else ir.Evaluate(ir.TensorLoad(storage, [integer(0)]))
        for access in accesses
    ]
    result = _inspect(_kernel([], [_unit(allocate(storage)), *map(_unit, body)]))
    tasks = [_ids(result, statement)[0] for statement in body]
    matching = [row for row in _pairs(result, root_only=True) if row[:2] == tuple(tasks)]
    assert bool(matching) == expected
    assert all(row[2] == 0 and row[3]["storage"].same_as(storage) for row in matching)


def test_address_definitions_do_not_become_opaque_memory_effects():
    storage, view = tensor("storage", [4], "int32", "global"), tensor("view", [2], "int32", "global")
    pointer = ir.Var("pointer", storage.data.ty)
    definitions = [
        allocate(storage),
        ir.Bind(pointer, storage.data),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
    ]
    result = _inspect(_kernel([], list(map(_unit, definitions))))
    assert not _pairs(result)
    assert all(not row["reads"] and not row["writes"] and not row["opaque"] for row in result["tasks"])


@pytest.mark.parametrize("origin", ["parameter", "same_fresh", "other_fresh"])
def test_opaque_pointer_consumers_retain_unbounded_dependencies(origin):
    storage, other = tensor("storage", [4], "int32", "global"), tensor("other", [4], "int32", "global")
    pointer = ir.Var("pointer", storage.data.ty)
    definitions, params = [allocate(storage)], []
    if origin == "parameter":
        params.append(pointer)
    else:
        if origin == "other_fresh":
            definitions.append(allocate(other))
        definitions.append(ir.Bind(pointer, (other if origin == "other_fresh" else storage).data))
    writer = ir.TensorStore(storage, [integer(0)], integer(1))
    opaque = ir.Call("tirx.call_extern", [ir.StringImm("opaque_consumer"), pointer], ty="int32")
    result = _inspect(_kernel(params, [*map(_unit, definitions), _unit(writer), _unit(ir.Evaluate(opaque))]))
    ids = (_ids(result, writer)[0], _ids(result, opaque)[0])
    assert any(row[:2] == ids for row in _pairs(result, root_only=True)) == (origin != "other_fresh")


@pytest.mark.parametrize("wrapper", ["if", "else", "assume", "begin", "end", "step"])
def test_control_headers_are_dependency_consumers(wrapper):
    state, output = tensor("state", [1], "int32", "global"), tensor("output", [1], "int32", "global")
    load = ir.TensorLoad(state, [integer(0)])
    writer = ir.TensorStore(state, [integer(0)], integer(2))
    child = _unit(ir.TensorStore(output, [integer(0)], integer(3)))
    if wrapper == "if":
        control = ir.If(load > 0, [child], None)
    elif wrapper == "else":
        control = ir.If(load > 0, [], [child])
    elif wrapper == "assume":
        control = attribute(load > 0, "tilelang_assume", integer(1), [child])
    else:
        control = _loop(ir.Var("i", "int32"), [child], **{wrapper: load})
    result = _inspect(_kernel([], [_unit(allocate(state)), _unit(allocate(output)), _unit(writer), control]))
    producer, consumer = _ids(result, writer)[0], _ids(result, load, header=True)[0]
    assert any(row[:2] == (producer, consumer) for row in _pairs(result, root_only=True))


@pytest.mark.parametrize("wrapper", ["if", "assume"])
def test_stage_order_keeps_a_write_before_its_lexical_header(wrapper):
    state = tensor("state", [1], "int32", "global")
    index = ir.Var("i", "int32")
    load = ir.TensorLoad(state, [integer(0)])
    writer = ir.TensorStore(state, [integer(0)], integer(2))
    child = _unit(ir.Evaluate(integer(0)))
    if wrapper == "if":
        scope = ir.If(load > 0, [child], None)
    else:
        scope = attribute(load > 0, "tilelang_assume", integer(1), [child])
    result = _inspect(_kernel([], [_unit(allocate(state)), _loop(index, [_unit(writer, stage=3), scope])]))
    producer, consumer = _ids(result, writer)[0], _ids(result, load, header=True)[0]
    pairs = _pairs(result, clock=index)
    assert any(row[:3] == (producer, consumer, 0) for row in pairs)
    assert not any(row[:3] == (consumer, producer, 0) for row in pairs)


@pytest.mark.parametrize("mutable", [False, True])
def test_separate_branch_evaluations_only_share_immutable_predicates(mutable):
    state = tensor("state", [1], "int32", "global")
    flag = tensor("flag", [1], "int32", "global") if mutable else ir.Var("flag", "bool")
    predicate = ir.TensorLoad(flag, [integer(0)]) > 0 if mutable else flag
    writer, reader = ir.TensorStore(state, [integer(0)], integer(3)), ir.TensorLoad(state, [integer(0)])
    result = _inspect(
        _kernel(
            [flag],
            [
                _unit(allocate(state)),
                ir.If(predicate, [_unit(writer)], None),
                ir.If(ir.prim.Not(predicate), [_unit(ir.Evaluate(reader))], None),
            ],
        )
    )
    ids = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:2] == ids for row in _pairs(result, root_only=True)) == mutable


@pytest.mark.parametrize("mutable", [False, True])
def test_opposite_branches_may_conflict_in_different_iterations(mutable):
    state = tensor("state", [1], "int32", "global")
    flag = tensor("flag", [1], "int32", "global") if mutable else ir.Var("flag", "bool")
    predicate = ir.TensorLoad(flag, [integer(0)]) > 0 if mutable else flag
    loop_var = ir.Var("i", "int32")
    writer, reader = ir.TensorStore(state, [integer(0)], integer(3)), ir.TensorLoad(state, [integer(0)])
    loop = _loop(loop_var, [ir.If(predicate, [_unit(writer)], [_unit(ir.Evaluate(reader))])])
    result = _inspect(_kernel([flag], [_unit(allocate(state)), loop]))
    ids = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:3] == (*ids, -1) for row in _pairs(result, clock=loop_var)) == mutable


def test_sibling_branch_lists_have_distinct_dependency_cache_entries():
    state = tensor("state", [2], "int32", "global")
    flag, index = ir.Var("flag", "bool"), ir.Var("i", "int32")
    writers = [ir.TensorStore(state, [integer(i)], integer(1)) for i in range(2)]
    reads = [ir.TensorLoad(state, [integer(i)]) for i in range(2)]
    branch = ir.If(flag, [_unit(writers[0]), _unit(ir.Evaluate(reads[0]))], [_unit(writers[1]), _unit(ir.Evaluate(reads[1]))])
    result = _inspect(_kernel([flag], [_unit(allocate(state)), _loop(index, [branch])]))
    rows = [scope for scope in result["scopes"] if "owner" in scope and isinstance(scope["owner"], ir.If)]
    assert len(rows) == 2
    for branch_id, scope in enumerate(rows):
        expected = [_ids(result, writers[branch_id])[0], _ids(result, reads[branch_id])[0]]
        assert any(
            list(pair) == expected and int(dependency["distance"]) == 0
            for dependency in scope["dependencies"]
            for pair in dependency["pairs"]
        )


@pytest.mark.parametrize("start,end,step", [(0, 16, 2), (7, 9, -2), (0, 8, 1), (3, 5, -1)])
def test_manual_ring_distance_counts_iterations_for_signed_steps(start, end, step):
    state = tensor("state", [2], "int32", "global")
    index = ir.Var("i", "int32")
    ordinal = (index - start) // step
    writer = ir.TensorStore(state, [ordinal % 2], integer(1))
    body = [_unit(writer)]
    if step < 0:
        body.append(ir.If(index <= (1 if step == -2 else 0), [_unit(ir.Break())], None))
    result = _inspect(_kernel([], [_unit(allocate(state)), _loop(index, body, begin=start, end=end, step=step)]), {state: 2})
    task = _ids(result, writer)[0]
    matching = [row for row in _pairs(result, clock=index) if row[:2] == (task, task)]
    assert matching and all(row[2] == 2 for row in matching)


@pytest.mark.parametrize("mode", ["no_conflict", "force", "contradictory", "same_only"])
def test_conflict_hint_polarity_and_cross_mode_are_native_operands(mode):
    left, right = tensor("left", [2], "int32", "global"), tensor("right", [2], "int32", "global")
    writer, reader = ir.TensorStore(left, [integer(0)], integer(1)), ir.TensorLoad(right, [integer(0)])
    hints = [[left, right, integer(0 if mode == "same_only" else -1), integer(mode in {"force", "same_only"}, "bool")]]
    if mode == "contradictory":
        hints.append([right, left, integer(-1), integer(True, "bool")])
    meta = annotation({"tl.root_conflict_hints": hints})
    result = _inspect(_kernel([], [_unit(allocate(left)), _unit(allocate(right)), meta, _unit(writer), _unit(ir.Evaluate(reader))]))
    ids = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:2] == ids for row in _pairs(result, root_only=True)) == (mode != "no_conflict")


@pytest.mark.parametrize("sid_pair,expected", [((0, 1), False), ((0, 0), True), ((None, 1), True)])
def test_ub_instances_are_separate_only_for_known_distinct_subcores(sid_pair, expected):
    state = tensor("state", [1], "int32", "shared.dyn")
    writer, reader = ir.TensorStore(state, [integer(0)], integer(1)), ir.TensorLoad(state, [integer(0)])
    cores = [5 if sid is None else (1 if sid == 0 else 4) for sid in sid_pair]
    result = _inspect(_kernel([], [_unit(allocate(state)), _unit(writer, cores[0]), _unit(ir.Evaluate(reader), cores[1])]))
    ids = (_ids(result, writer)[0], _ids(result, reader)[0])
    assert any(row[:2] == ids for row in _pairs(result, root_only=True)) == expected


def test_storage_owner_exclusion_excludes_the_owners_own_header():
    state = tensor("state", [1], "int32", "global")
    left, right = ir.Var("left", "int32"), ir.Var("right", "int32")
    bound = ir.TensorLoad(state, [integer(0)])
    first, second = ir.TensorStore(state, [integer(0)], integer(2)), ir.TensorStore(state, [integer(0)], integer(3))
    result = _inspect(
        _kernel([], [_unit(allocate(state)), _loop(left, [_unit(first)]), _loop(right, [_unit(second)], end=bound)]),
        owners={state: [left, right]},
    )
    first_id, second_id, header = _ids(result, first)[0], _ids(result, second)[0], _ids(result, bound, header=True)[0]
    pairs = _pairs(result, root_only=True)
    assert any(row[:2] == (first_id, second_id) and row[3]["owner_exclusion"] for row in pairs)
    assert any(row[:2] == (first_id, header) and not row[3]["owner_exclusion"] for row in pairs)


def test_mutable_index_reads_are_independent_but_ssa_indices_are_shared():
    state, indices = tensor("state", [8], "int32", "global"), tensor("indices", [1], "int32", "global")
    loaded = ir.TensorLoad(indices, [integer(0)])
    assert _conflict(subregion(state, [loaded], [1]), subregion(state, [loaded + 1], [1]))
    index = ir.Var("index", "int32")
    assert not _conflict(subregion(state, [index], [1]), subregion(state, [index + 1], [1]))


def test_half_open_endpoints_do_not_wrap_at_the_int32_boundary():
    state = tensor("state", [integer(2**31, "int64")], "int32", "global")
    assert not _conflict(subregion(state, [2**31 - 1], [1]), subregion(state, [2**31 - 2], [1]))


@pytest.mark.parametrize("leading_stride,expected", [(256, False), (0, True)])
def test_unknown_l0_pitch_cannot_hide_aliased_version_axes(leading_stride, expected):
    state = strided_tensor("state", [2, 16, 16], [leading_stride, 16, 1], 0, "float16", "shared.l0a")
    assert _conflict(subregion(state, [0, 0, 0], [1, 1, 1]), subregion(state, [1, 8, 8], [1, 1, 1])) == expected


def test_unknown_l0_pitch_keeps_matrix_tiles_conflicting():
    state = tensor("state", [16, 16], "float16", "shared.l0a")
    left, right = subregion(state, [0, 0], [1, 1]), subregion(state, [8, 8], [1, 1])
    assert _conflict(left, right)
    assert not _conflict(left, right, pitches=(integer(16), integer(16)))
