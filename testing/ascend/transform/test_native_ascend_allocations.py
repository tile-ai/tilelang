"""Native Ascend storage versions, completion edges and arena addresses."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx

from tilelang.testing.packed import compile_packed_function
from tvm_tirx.tirx.layout import S, TileLayout


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape=(64,), dtype="float32", scope="shared.dyn", layout=None, align=0):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout, align=align))


def allocate(value, attrs=None):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty], attrs=attrs))


def declare(value, pointer):
    return ir.Bind(value, ir.Call("tirx.decl_tensor", [pointer], ty_args=[value.ty]))


def annotation(fields):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def attribute(key, value, body, node=None):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def objects(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def calls(value, name):
    return [call for call in objects(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == name]


def metadata(value):
    return [
        tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
        for call in calls(value, "tl.annotation")
    ]


def constant(value):
    return int(sym.Analyzer().simplify(value).value)


def apply(name, body, params=(), arguments=(), attrs=None, check_input=True):
    before = tirx.Function(list(params), body, attrs=attrs)
    if check_input:
        assert tirx.analysis.check_well_formed(before)
        for call in objects(before, ir.Call):
            call.validate()
    saved = ir.save_json(before)
    transform = tvm_ffi.get_global_func("tl.transform." + name)(*arguments)
    after = transform(ir.IRModule({"main": before}))["main"]
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    for call in objects(after, ir.Call):
        call.validate()
    return after


def pointer(value, indices=None, extent=64, mask=1):
    indices = [integer(0)] if indices is None else [integer(i) if isinstance(i, int) else i for i in indices]
    extent = integer(extent) if isinstance(extent, int) else extent
    return ir.Call("tl.access_ptr", [ir.TensorLoad(value, indices), extent, integer(mask)])


def intrinsic(name, *args):
    args = [ir.StringImm(value) if isinstance(value, str) else integer(value) if isinstance(value, int) else value for value in args]
    return ir.Evaluate(ir.Call("tl.ascend_" + name, args, ty_args=[ir.PrimType("void")]))


def flags(pipe, index=0):
    return [intrinsic(kind + "_flag", pipe, index) for kind in ("set", "wait")]


def load(gm, ub):
    return intrinsic("copy_gm_to_ubuf", pointer(ub, mask=2), pointer(gm), 0, 1, 256, 0, 0, 0, 0, 0, 0)


def store(ub, gm):
    return intrinsic("copy_ubuf_to_gm", pointer(gm, mask=2), pointer(ub), 0, 1, 256, 0, 0, 0)


def contract(value):
    result = {}
    for fields in metadata(value):
        if "tl.buffer_alias_map" in fields:
            for var, compatible in fields["tl.buffer_alias_map"].items():
                assert var not in result
                result[var] = list(compatible)
    return result


def may_alias(result, a, b):
    aliases = contract(result)
    return any(other.same_as(b) for other in aliases[a])


def definitions(value):
    return {binding.var: binding.value for binding in objects(value, ir.Bind)}


def address(value, bindings):
    """Read concrete arena provenance without compiler alias-analysis helpers."""
    if isinstance(value, ir.Var):
        if value not in bindings:
            return value, 0
        binding = bindings[value]
        if isinstance(binding, ir.Call) and binding.op.name == "tirx.alloc_tensor":
            return value, 0
        return address(binding, bindings)
    assert isinstance(value, ir.Call), value
    if value.op.name in ("tirx.tensor_data_ptr", "tirx.decl_tensor", "tirx.reinterpret"):
        return address(value.args[0], bindings)
    assert value.op.name == "tirx.ptr_byte_offset", value
    root, offset = address(value.args[0], bindings)
    return root, offset + constant(value.args[1])


@pytest.mark.parametrize("release", ["none", "acquire-only", "MTE3_MTE2", "PIPE_ALL"])
def test_dma_read_requires_completion_before_reuse(release):
    gm = tensor("gm", scope="global")
    a, b = tensor("a"), tensor("b")
    between = []
    if release == "acquire-only":
        between = [intrinsic("wait_flag", "MTE3_MTE2", 0)]
    elif release == "MTE3_MTE2":
        between = flags(release)
    elif release == "PIPE_ALL":
        between = [intrinsic("pipe_barrier", release)]
    body = [
        allocate(a),
        allocate(b),
        load(gm, a),
        *flags("MTE2_MTE3"),
        store(a, gm),
        *between,
        load(gm, b),
        *flags("MTE2_MTE3"),
        store(b, gm),
    ]
    result = apply("InferBufferAliases", body, [gm])
    assert may_alias(result, a, b) == (release in ("MTE3_MTE2", "PIPE_ALL"))


@pytest.mark.parametrize("release_pipe", ["S_MTE2", "V_MTE2"])
def test_pad_value_uses_scalar_pipe_inside_vector_block(release_pipe):
    gm, a, b = tensor("gm", scope="global"), tensor("a"), tensor("b")
    inside = [load(gm, a), *flags("MTE2_S"), intrinsic("set_copy_pad_value", ir.TensorLoad(a, [0])), *flags(release_pipe), load(gm, b)]
    result = apply("InferBufferAliases", [allocate(a), allocate(b), attribute("tl.sblock", {"name": "VECTOR"}, inside)], [gm])
    assert may_alias(result, a, b) == (release_pipe == "S_MTE2")


@pytest.mark.parametrize("release_pipe", ["PIPE_MTE1", "PIPE_FIX"])
def test_cross_core_handshake_releases_only_its_local_pipe(release_pipe):
    gm = tensor("gm", scope="global")
    a, b = tensor("a", scope="shared.l1"), tensor("b", scope="shared.l1")
    l0a = tensor("l0a", scope="shared.l0a")

    def cube_copy(destination):
        return intrinsic("copy_gm_to_cbuf", pointer(destination, mask=2), pointer(gm), 0, 16, 0, 4, 16, 0, 0, 0, 4)

    first = attribute(
        "tl.sblock",
        {"name": "CUBE"},
        [
            cube_copy(a),
            *flags("MTE2_MTE1"),
            intrinsic("load_cbuf_to_ca", pointer(l0a, mask=2), pointer(a), 0, 1, 0, 0, 0, 0, 0),
            intrinsic("cross_core_set_flag", 4, release_pipe, 0),
        ],
    )
    vector = attribute(
        "tl.sblock",
        {"name": "VECTOR"},
        [intrinsic("cross_core_wait_flag", 4, "PIPE_MTE3", 0), intrinsic("cross_core_set_flag", 4, "PIPE_MTE3", 1)],
    )
    last = attribute("tl.sblock", {"name": "CUBE"}, [intrinsic("cross_core_wait_flag", 4, "PIPE_MTE2", 1), cube_copy(b)])
    result = apply("InferBufferAliases", [allocate(a), allocate(b), allocate(l0a), first, vector, last], [gm])
    assert may_alias(result, a, b) == (release_pipe == "PIPE_MTE1")


def test_same_named_symbol_is_not_an_inferred_pipeline_iteration():
    gm, a, b = tensor("gm", scope="global"), tensor("a"), tensor("b")
    induction, parameter = ir.Var("i", "int32"), ir.Var("i", "int32")
    release = intrinsic("set_flag", "MTE3_MTE2", induction)
    acquire = intrinsic("wait_flag", "MTE3_MTE2", parameter)
    loop = ir.For(induction, integer(0), integer(1), integer(1), ir.ForKind.DEFAULT, [release])
    result = apply("InferBufferAliases", [allocate(a), allocate(b), store(a, gm), loop, acquire, load(gm, b)], [gm, parameter])
    assert not may_alias(result, a, b)


@pytest.mark.parametrize("begin,end,step", [(3, 4, 1), (3, 5, 2)])
def test_event_ids_follow_exact_native_begin_end_step(begin, end, step):
    gm, a, b = tensor("gm", scope="global"), tensor("a"), tensor("b")
    induction = ir.Var("i", "int32")
    loop = ir.For(
        induction, integer(begin), integer(end), integer(step), ir.ForKind.DEFAULT, [intrinsic("set_flag", "MTE3_MTE2", induction)]
    )
    # The first actual induction value releases the matching wait, without an
    # invented zero-origin ordinal replacing the source event id.
    result = apply(
        "InferBufferAliases", [allocate(a), allocate(b), store(a, gm), loop, intrinsic("wait_flag", "MTE3_MTE2", begin), load(gm, b)], [gm]
    )
    assert may_alias(result, a, b)


@pytest.mark.parametrize("case", ["zero-trip", "unknown-loop", "unknown-if", "wrapping-loop", "negative-step-empty"])
def test_unknown_or_unexecuted_release_never_proves_reuse(case):
    gm, a, b = tensor("gm", scope="global"), tensor("a"), tensor("b")
    n = ir.Var("n", "int32")
    params = [gm, n]
    release = intrinsic("set_flag", "MTE3_MTE2", 0)
    i = ir.Var("i", "int32")
    if case == "unknown-if":
        release = ir.If(n > 0, [release], None)
    elif case == "negative-step-empty":
        # Native For always tests i < end; a negative step does not invert it.
        release = ir.For(i, integer(5), integer(2), integer(-2), ir.ForKind.DEFAULT, [release])
    elif case == "wrapping-loop":
        i = ir.Var("i", "int8")
        release = ir.For(i, integer(126, "int8"), integer(127, "int8"), integer(2, "int8"), ir.ForKind.DEFAULT, [release])
    else:
        release = ir.For(i, integer(0), integer(0) if case == "zero-trip" else n, integer(1), ir.ForKind.DEFAULT, [release])
    result = apply(
        "InferBufferAliases", [allocate(a), allocate(b), store(a, gm), release, intrinsic("wait_flag", "MTE3_MTE2", 0), load(gm, b)], params
    )
    assert not may_alias(result, a, b)


def test_alias_load_keeps_the_allocation_lifetime_live():
    gm, a, b = tensor("gm", scope="global"), tensor("a"), tensor("b")
    view = tensor("view", (32,), scope="shared.dyn")
    binding = declare(view, ir.Call("tirx.ptr_byte_offset", [a.data, integer(4)]))
    body = [allocate(a), allocate(b), binding, load(gm, a), *flags("MTE2_MTE3"), store(view, gm), *flags("V_MTE2"), load(gm, b)]
    result = apply("InferBufferAliases", body, [gm])
    assert not may_alias(result, a, b)
    assert view not in contract(result)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_arena_packing_requires_pairwise_compatibility(disable_reuse):
    a, b, c = [tensor(name, (size,), "int32") for name, size in (("a", 64), ("b", 32), ("c", 32))]
    aliases = {a: [b, c], b: [a], c: [a]}
    body = [allocate(value) for value in (a, b, c)] + [annotation({"tl.buffer_alias_map": aliases})]
    body.extend(ir.TensorStore(value, [0], integer(index)) for index, value in enumerate((a, b, c)))
    result = apply("MergeUBAllocations", body, arguments=(16, disable_reuse))
    assert not contract(result)
    bindings = definitions(result)
    (arena,) = [v for v, call in bindings.items() if call.op.name == "tirx.alloc_tensor"]
    pairs = [address(value, bindings) for value in (a, b, c)]
    assert all(root.same_as(arena) for root, _ in pairs)
    offsets = [offset for _, offset in pairs]
    sizes = [256, 128, 128]
    assert offsets[1] + sizes[1] <= offsets[2] or offsets[2] + sizes[2] <= offsets[1]
    if disable_reuse:
        for left, right in ((0, 1), (0, 2)):
            assert offsets[left] + sizes[left] <= offsets[right] or offsets[right] + sizes[right] <= offsets[left]
    else:
        assert any(offsets[0] < offsets[i] + sizes[i] and offsets[i] < offsets[0] + sizes[0] for i in (1, 2))
    for index, value in enumerate((a, b, c)):
        assert bindings[value].ty.same_as(value.ty)
        assert offsets[index] >= 0 and offsets[index] + sizes[index] <= constant(arena.ty.shape[0])


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_private_aiv_arenas_share_addresses_but_common_storage_is_disjoint(disable_reuse):
    common, first, second = [tensor(name, (size,), "int32") for name, size in (("common", 32), ("first", 128), ("second", 64))]
    values = (common, first, second)
    fields = {"tl.buffer_alias_map": {value: [] for value in values}, "tl.aiv_storage_map": {first: 0, second: 1}}
    result = apply("MergeUBAllocations", [*[allocate(v) for v in values], annotation(fields)], arguments=(16, disable_reuse))
    bindings = definitions(result)
    arena, base = address(common, bindings)
    offsets = [address(value, bindings)[1] for value in values]
    size = constant(arena.ty.shape[0])
    assert size == (32 + max(128, 64)) * 4
    for index, extent in enumerate((32, 128, 64)):
        assert offsets[index] >= 0 and offsets[index] + extent * 4 <= size
        if index:
            assert base + 128 <= offsets[index] or offsets[index] + extent * 4 <= base
    assert offsets[1] < offsets[2] + 256 and offsets[2] < offsets[1] + 512


@pytest.mark.parametrize(
    "scope,alignment", [("shared.ss", 32), ("shared.l1", 512), ("shared.l0a", 512), ("shared.l0b", 512), ("shared.l0c", 1024)]
)
def test_arena_obeys_each_hardware_memory_scope(scope, alignment):
    a, b = tensor("a", (2,), "uint32", scope, align=alignment), tensor("b", (2,), "uint32", scope, align=alignment)
    compatible = {a: [b], b: [a]} if scope == "shared.ss" else {a: [], b: []}
    result = apply("MergeUBAllocations", [allocate(a), allocate(b), annotation({"tl.buffer_alias_map": compatible})], arguments=(16, False))
    bindings = definitions(result)
    offsets = [address(value, bindings)[1] for value in (a, b)]
    assert sorted(offsets) == [0, alignment]
    (arena,) = [v for v, call in bindings.items() if call.op.name == "tirx.alloc_tensor"]
    assert arena.ty.scope.name == scope + ".dyn"
    assert int(arena.ty.data_alignment) >= alignment


def test_merge_preserves_view_layout_dtype_and_byte_offset_exactly_once():
    a = tensor("a", (16,), "int16", align=64)
    b = tensor("b", (3,), "uint8")
    view = tensor("view", (2, 3), "uint8", layout=TileLayout(S[(2, 3) : (8, 2)] + 1))
    aliased = ir.Call("tirx.reinterpret", [a.data], ty=view.data.ty)
    body = [
        allocate(a),
        allocate(b),
        declare(view, ir.Call("tirx.ptr_byte_offset", [aliased, integer(3)])),
        annotation({"tl.buffer_alias_map": {a: [], b: []}}),
        ir.TensorStore(view, [1, 2], integer(42, "uint8")),
    ]
    result = apply("MergeUBAllocations", body, arguments=(16, False))
    bindings = definitions(result)
    root_a, offset_a = address(a, bindings)
    root_view, offset_view = address(view, bindings)
    assert root_a.same_as(root_view) and offset_view == offset_a + 3
    assert bindings[a].ty.same_as(a.ty) and bindings[view].ty.same_as(view.ty)
    assert [constant(index) for index in objects(result, ir.TensorStore)[0].indices] == [1, 2]
    assert offset_a % 64 == 0


@pytest.mark.parametrize("placement", ["marker", "block", "loop"])
def test_manual_versions_normalize_at_native_metadata_site(placement):
    storage = tensor("storage", (2, 3), "int32")
    fields = {"tl.manual_multi_buffer": {storage: 2}, "keep": "user-tag"}
    body = [allocate(storage)]
    if placement == "marker":
        body.extend([annotation(fields), ir.TensorStore(storage, [1, 2], integer(7))])
    else:
        key = "tl.sblock" if placement == "block" else "tl.loop.annotations"
        body.append(attribute(key, fields, [ir.TensorStore(storage, [1, 2], integer(7))]))
    result = apply("NormalizeBufferVersion", body)
    found = metadata(result)
    for region in objects(result, ir.RegionStmt):
        if region.op.name == "tl.attr":
            decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
            found.append(decoded["value"])
    (actual,) = [value for value in found if "tl.buffer_version" in value]
    assert "tl.manual_multi_buffer" not in actual
    assert actual["keep"] == "user-tag" and int(actual["tl.buffer_version"][storage]) == 2


@pytest.mark.parametrize("count", [0, -1, 1 << 32])
def test_manual_versions_require_positive_int32_count(count):
    storage = tensor("storage")
    with pytest.raises(Exception, match="positive int32"):
        apply("NormalizeBufferVersion", [allocate(storage), annotation({"tl.manual_multi_buffer": {storage: count}})])


def test_frontend_cannot_supply_compiler_physical_version_metadata():
    storage = tensor("storage")
    with pytest.raises(Exception, match="compiler-internal"):
        apply("NormalizeBufferVersion", [allocate(storage), annotation({"tl.buffer_version": {storage: 2}})])


@pytest.mark.parametrize("dtype,extent", [("int8", 5), ("int16", 3), ("int32", 3), ("float4_e2m1fn", 6)])
def test_each_compact_version_maps_to_its_own_32_byte_slot(dtype, extent):
    storage = tensor("storage", (2, extent), dtype)
    reads = [ir.Evaluate(ir.TensorLoad(storage, [version, index])) for version in range(2) for index in range(extent)]
    result = apply("RewriteAscendBufferVersionLayout", [allocate(storage), annotation({"tl.buffer_version": {storage: 2}}), *reads])
    (physical,) = [binding.var for binding in objects(result, ir.Bind)]
    bits = int(storage.ty.dtype.dtype.bits)
    assert constant(physical.ty.shape[0]) * bits == 2 * 256
    loads = objects(result, ir.TensorLoad)
    assert [constant(load.indices[0]) * bits for load in loads] == [
        version * 256 + index * bits for version in range(2) for index in range(extent)
    ]
    assert all(load.source.same_as(physical) for load in loads)
    assert not any("tl.buffer_version" in fields for fields in metadata(result))


def test_version_rewrite_includes_nonzero_layout_origin_and_noncompact_stride():
    storage = tensor("storage", (2, 3), "int32", layout=TileLayout(S[(2, 3) : (5, 1)] + 2))
    reads = [ir.Evaluate(ir.TensorLoad(storage, [version, index])) for version in range(2) for index in range(3)]
    result = apply(
        "RewriteAscendBufferVersionLayout",
        [allocate(storage), annotation({"tl.buffer_version": {storage: 2}}), *reads, ir.Evaluate(storage.data)],
    )
    loads = objects(result, ir.TensorLoad)
    assert [constant(load.indices[0]) for load in loads] == [2, 3, 4, 10, 11, 12]
    (physical,) = [binding.var for binding in objects(result, ir.Bind)]
    assert constant(physical.ty.shape[0]) == 18
    pointer_calls = calls(result, "tirx.ptr_byte_offset")
    assert constant(pointer_calls[-1].args[1]) == 0


def test_cross_dtype_alias_uses_one_storage_bit_mapping():
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (10,), "int16")
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    reads = [ir.Evaluate(ir.TensorLoad(view, [index])) for index in range(10)]
    body = [
        allocate(storage),
        declare(view, ir.Call("tirx.ptr_byte_offset", [base, integer(4)], ty=base.ty)),
        annotation({"tl.buffer_version": {storage: 2}}),
        *reads,
    ]
    result = apply("RewriteAscendBufferVersionLayout", body)
    loads = objects(result, ir.TensorLoad)
    expected = []
    for index in range(10):
        old_bits = 4 * 8 + index * 16
        version, inner = divmod(old_bits, 3 * 32)
        expected.append((version * 256 + inner) // 16)
    assert [constant(load.indices[0]) for load in loads] == expected
    bindings = definitions(result)
    root, offset = address(loads[0].source, bindings)
    assert offset == 0 and root.name == "storage"


def test_pointer_extent_cannot_cross_a_padded_version_boundary():
    storage = tensor("storage", (2, 3), "int32")
    with pytest.raises(Exception, match="crosses a 32-byte-aligned version slot"):
        apply(
            "RewriteAscendBufferVersionLayout",
            [allocate(storage), annotation({"tl.buffer_version": {storage: 2}}), ir.Evaluate(pointer(storage, [0, 2], extent=2))],
        )


@pytest.mark.parametrize("case", ["already-aligned", "single", "flat", "L1"])
def test_nonapplicable_version_shapes_keep_existing_tensor_type(case):
    shape = (2, 8) if case == "already-aligned" else (6,) if case == "flat" else (1, 3) if case == "single" else (2, 3)
    storage = tensor("storage", shape, "int32", "shared.l1" if case == "L1" else "shared.dyn")
    count = 1 if case == "single" else 2
    result = apply("RewriteAscendBufferVersionLayout", [allocate(storage), annotation({"tl.buffer_version": {storage: count}})])
    (binding,) = objects(result, ir.Bind)
    assert binding.var.same_as(storage) and binding.var.ty.same_as(storage.ty)
    assert not any("tl.buffer_version" in fields for fields in metadata(result))


def unit(statement, stage=0, core=5, guard=None):
    task = {"core_mask": integer(core, "int64")} if core else {}
    if isinstance(statement, ir.Bind):
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"stage": integer(stage, "int64"), "task": task})
        call = ir.Call(
            "tl.ascend.scheduled_bind",
            [statement.value, payload],
            attrs={"schema_version": 1, "operand_count": 1, "annotation_schema": schema},
        )
        result = ir.Bind(statement.var, call)
    else:
        if not isinstance(statement, ir.For):
            statement = attribute("tl.ascend_task", integer(1), [statement], node=task)
        result = attribute("tl.schedule_unit", integer(1), [statement], node={"stage": integer(stage, "int64")})
    return result if guard is None else ir.If(guard, [result], None)


def scheduled(body, buffers=(), fields=None):
    sequence = [unit(allocate(value)) for value in buffers]
    if fields:
        sequence.append(annotation(fields))
    sequence.extend(body)
    return attribute("tl.sblock", {"name": "tilelang_root", "no_realize": True}, sequence)


def loop_claims(loop):
    claims = list(loop.annotations.get("multi_buffer_eligible", []))
    for statement in loop.body.seq:
        if isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call) and statement.value.op.name == "tl.annotation":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(statement.value.args[0], statement.value.attrs["annotation_schema"])
            claims.extend(fields.get("multi_buffer_eligible", []))
    return claims


@pytest.mark.parametrize(
    "case,expected",
    [
        ("write-first", True),
        ("read-first", False),
        ("same-guard", True),
        ("disjoint-guards", False),
        ("exhaustive-writes", True),
        ("manual-stage-order", True),
        ("vf-write-first", True),
        ("vf-read-first", False),
    ],
)
def test_eligibility_keeps_write_first_ownership_through_native_scopes(case, expected):
    ub, out = tensor("ub", (1,), "int32"), tensor("out", (4,), "int32", "global")
    enabled, induction = ir.Var("enabled", "bool"), ir.Var("i", "int32")
    write, read = ir.TensorStore(ub, [0], induction), ir.TensorStore(out, [induction], ir.TensorLoad(ub, [0]))
    if case.startswith("vf-"):
        body = [unit(attribute("tl.sblock", {"name": "SIMD_VF", "no_realize": True}, [write, read] if expected else [read, write]))]
    elif case == "same-guard":
        body = [unit(write, guard=enabled), unit(read, guard=enabled)]
    elif case == "disjoint-guards":
        body = [unit(write, guard=enabled), unit(read, guard=~enabled)]
    elif case == "exhaustive-writes":
        body = [unit(write, guard=enabled), unit(write, guard=~enabled), unit(read)]
    elif case == "manual-stage-order":
        body = [unit(read, stage=1), unit(write, stage=0)]
    elif case == "read-first":
        body = [unit(read), unit(write)]
    else:
        body = [unit(write), unit(read)]
    loop = ir.For(induction, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body)
    result = apply("AnnotateMultiBufferEligible", [scheduled([unit(loop)], [ub])], [out, enabled])
    (owner,) = objects(result, ir.For)
    assert any(var.same_as(ub) for var in loop_claims(owner)) == expected


def test_eligibility_does_not_equate_separate_mutable_condition_reads():
    ub, out, gate = tensor("ub", (1,), "int32"), tensor("out", (4,), "int32", "global"), tensor("gate", (1,), "int32", "global")
    i = ir.Var("i", "int32")
    condition = ir.TensorLoad(gate, [0]) != 0
    body = [
        unit(ir.TensorStore(ub, [0], i), guard=condition),
        unit(ir.TensorStore(gate, [0], integer(1))),
        unit(ir.TensorStore(out, [i], ir.TensorLoad(ub, [0])), guard=condition),
    ]
    loop = ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body)
    result = apply("AnnotateMultiBufferEligible", [scheduled([unit(loop)], [ub])], [out, gate])
    assert not any(var.same_as(ub) for var in loop_claims(objects(result, ir.For)[0]))


def test_nested_alias_read_promotes_the_complete_storage_owner():
    base, alias = tensor("base", (4,), "int32"), tensor("alias", (2, 2), "int32")
    out = tensor("out", (4,), "int32", "global")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    read = unit(ir.TensorStore(out, [inner], ir.TensorLoad(alias, [0, inner])))
    child = ir.For(inner, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [read])
    parent = ir.For(outer, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, [unit(ir.TensorStore(base, [0], outer)), unit(child)])
    result = apply("AnnotateMultiBufferEligible", [scheduled([unit(declare(alias, base.data)), unit(parent)], [base])], [out])
    owners = [loop.var for loop in objects(result, ir.For) if any(var.same_as(base) for var in loop_claims(loop))]
    assert len(owners) == 1 and owners[0].same_as(outer)


@pytest.mark.parametrize("manual", [False, True])
def test_explicit_claim_and_manual_version_exclude_other_automatic_owners(manual):
    ub, out = tensor("ub", (2,), "int32"), tensor("out", (4,), "int32", "global")
    loops = []
    for index in range(2):
        i = ir.Var("i" + str(index), "int32")
        body = [unit(ir.TensorStore(ub, [0], i)), unit(ir.TensorStore(out, [i], ir.TensorLoad(ub, [0])))]
        if index == 0:
            body.append(annotation({"multi_buffer_eligible": [ub]}))
        loops.append(unit(ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body)))
    fields = {"tl.manual_multi_buffer": {ub: 2}} if manual else {}
    result = apply("AnnotateMultiBufferEligible", [scheduled(loops, [ub], fields)], [out])
    claimed = [any(var.same_as(ub) for var in loop_claims(loop)) for loop in objects(result, ir.For)]
    assert claimed == ([False, False] if manual else [True, False])


@pytest.mark.parametrize("mode,expected", [(None, False), ("auto", True), ("iteration", True), ("counter", True)])
def test_single_version_opt_out_is_distinct_from_an_explicit_mode(mode, expected):
    ub, out = tensor("ub", (1,), "int32"), tensor("out", (4,), "int32", "global")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        [unit(ir.TensorStore(ub, [0], i)), unit(ir.TensorStore(out, [i], ir.TensorLoad(ub, [0])))],
    )
    fields = {"tl.buffer_versions_map": {ub: 1}}
    if mode:
        fields["tl.buffer_version_mode"] = {ub: mode}
    result = apply("AnnotateMultiBufferEligible", [scheduled([unit(loop)], [ub], fields)], [out])
    assert any(var.same_as(ub) for var in loop_claims(objects(result, ir.For)[0])) == expected


def compile_cpu(function, tmp_path):
    target = tvm_tirx.target.Target("c")
    function = function.with_attr("global_symbol", "main").with_attr("target", target)
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("c")(ir.IRModule({"main": function}))
    (tmp_path / "lowered.json").write_text(ir.save_json(module))
    return compile_packed_function(module)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_merged_arena_executes_values_on_cpu(disable_reuse, tmp_path):
    a, b = tensor("a", (4,), "int32"), tensor("b", (4,), "int32")
    out = tensor("out", (8,), "int32", "global")
    body = [allocate(a), allocate(b), annotation({"tl.buffer_alias_map": {a: [b], b: [a]}})]
    for value, output_offset in ((a, 0), (b, 4)):
        body.extend(ir.TensorStore(value, [index], integer(10 + output_offset + index)) for index in range(4))
        body.extend(ir.TensorStore(out, [output_offset + index], ir.TensorLoad(value, [index])) for index in range(4))
    result = apply("MergeUBAllocations", body, [out], arguments=(16, disable_reuse))
    function = compile_cpu(result, tmp_path)
    array = tvm_tirx.runtime.tensor(np.full(8, -1, dtype="int32"))
    function(array)
    np.testing.assert_array_equal(array.numpy(), np.arange(10, 18, dtype="int32"))


def test_padded_versions_and_flat_alias_execute_values_on_cpu(tmp_path):
    storage = tensor("storage", (2, 3), "int32")
    alias = tensor("alias", (6,), "int32")
    out = tensor("out", (6,), "int32", "global")
    body = [allocate(storage), declare(alias, storage.data), annotation({"tl.buffer_version": {storage: 2}})]
    body.extend(
        ir.TensorStore(storage, [version, index], integer(100 + version * 10 + index)) for version in range(2) for index in range(3)
    )
    body.extend(ir.TensorStore(out, [index], ir.TensorLoad(alias, [index])) for index in range(6))
    result = apply("RewriteAscendBufferVersionLayout", body, [out])
    function = compile_cpu(result, tmp_path)
    array = tvm_tirx.runtime.tensor(np.full(6, -1, dtype="int32"))
    function(array)
    np.testing.assert_array_equal(array.numpy(), np.array([100, 101, 102, 110, 111, 112], dtype="int32"))


def block_fields(value):
    result = []
    for region in objects(value, ir.RegionStmt):
        if region.op.name == "tl.attr":
            decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
            if decoded["key"] == "tl.sblock":
                result.append(decoded["value"])
    return result


def test_manual_version_reads_the_real_sblock_annotation_envelope():
    storage = tensor("storage", (2, 3), "int32")
    block = attribute(
        "tl.sblock",
        {"name": "tilelang_root", "no_realize": True, "annotations": {"tl.manual_multi_buffer": {storage: 2}, "retained": "tag"}},
        [],
    )
    normalized = apply("NormalizeBufferVersion", [allocate(storage), block])
    (fields,) = block_fields(normalized)
    assert fields["name"] == "tilelang_root"
    assert fields["annotations"]["retained"] == "tag"
    assert int(fields["annotations"]["tl.buffer_version"][storage]) == 2
    rewritten = apply("RewriteAscendBufferVersionLayout", normalized.body)
    assert constant(objects(rewritten, ir.Bind)[0].var.ty.shape[0]) == 16
    assert "tl.buffer_version" not in block_fields(rewritten)[0]["annotations"]


def test_nested_frontend_physical_version_is_rejected():
    storage = tensor("storage")
    block = attribute("tl.sblock", {"name": "VECTOR", "annotations": {"tl.buffer_version": {storage: 2}}}, [])
    with pytest.raises(Exception, match="compiler-internal"):
        apply("NormalizeBufferVersion", [allocate(storage), block])


def test_alias_inference_replaces_nested_sblock_contract():
    a, b = tensor("a", (1,), "int32"), tensor("b", (1,), "int32")
    block = attribute(
        "tl.sblock",
        {"name": "VECTOR", "annotations": {"tl.buffer_alias_map": {a: [b], b: [a]}, "retained": 7}},
        [ir.TensorStore(a, [0], integer(1)), ir.TensorStore(b, [0], integer(2))],
    )
    result = apply("InferBufferAliases", [allocate(a), allocate(b), block])
    assert not may_alias(result, a, b)
    (fields,) = block_fields(result)
    assert "tl.buffer_alias_map" not in fields["annotations"]
    assert fields["annotations"]["retained"] == 7


def test_arena_consumes_nested_sblock_alias_contract():
    a, b = tensor("a", (1,), "int32"), tensor("b", (1,), "int32")
    block = attribute("tl.sblock", {"name": "VECTOR", "annotations": {"tl.buffer_alias_map": {a: [b], b: [a]}, "retained": "x"}}, [])
    result = apply("MergeUBAllocations", [allocate(a), allocate(b), block], arguments=(16, False))
    bindings = definitions(result)
    root_a, offset_a = address(a, bindings)
    root_b, offset_b = address(b, bindings)
    assert root_a.same_as(root_b) and offset_a == offset_b
    (fields,) = block_fields(result)
    assert fields["annotations"]["retained"] == "x" and "tl.buffer_alias_map" not in fields["annotations"]


@pytest.mark.parametrize("extra_workspace", [False, True])
def test_shaped_alias_versions_a_flat_allocation_only_when_it_covers_the_ring(extra_workspace):
    owner = tensor("owner", (10 if extra_workspace else 6,), "int32")
    alias = tensor("alias", (2, 3), "int32")
    body = [
        allocate(owner),
        declare(alias, owner.data),
        annotation({"tl.buffer_version": {alias: 2}}),
        ir.Evaluate(ir.TensorLoad(alias, [1, 0])),
    ]
    result = apply("RewriteAscendBufferVersionLayout", body)
    (load_node,) = objects(result, ir.TensorLoad)
    if extra_workspace:
        assert load_node.source.same_as(alias)
        assert len(load_node.indices) == 2
        assert objects(result, ir.Bind)[0].var.same_as(owner)
    else:
        assert constant(load_node.indices[0]) == 8
        assert constant(objects(result, ir.Bind)[0].var.ty.shape[0]) == 16


@pytest.mark.parametrize("policy", ["opt-out", "manual", "explicit"])
def test_eligibility_policy_resolves_declared_alias_to_its_allocation(policy):
    ub, alias = tensor("ub", (1,), "int32"), tensor("alias", (1,), "int32")
    out = tensor("out", (4,), "int32", "global")
    i = ir.Var("i", "int32")
    body = [unit(ir.TensorStore(ub, [0], i)), unit(ir.TensorStore(out, [i], ir.TensorLoad(alias, [0])))]
    fields = {}
    if policy == "explicit":
        body.append(annotation({"multi_buffer_eligible": [alias]}))
    elif policy == "opt-out":
        fields = {"tl.buffer_versions_map": {alias: 1}}
    else:
        fields = {"tl.manual_multi_buffer": {alias: 2}}
    loop = unit(ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body))
    result = apply("AnnotateMultiBufferEligible", [scheduled([unit(declare(alias, ub.data)), annotation(fields), loop], [ub])], [out])
    claims = loop_claims(objects(result, ir.For)[0])
    assert any(var.same_as(ub) for var in claims) == (policy == "explicit")
    assert not any(var.same_as(alias) for var in claims)


def test_conflicting_version_counts_on_two_aliases_are_not_last_writer_wins():
    ub, alias = tensor("ub", (1,), "int32"), tensor("alias", (1,), "int32")
    fields = {"tl.buffer_versions_map": {ub: 2, alias: 3}}
    body = [scheduled([unit(declare(alias, ub.data)), annotation(fields)], [ub])]
    with pytest.raises(Exception, match="Conflicting buffer version policies"):
        apply("AnnotateMultiBufferEligible", body)


def test_equal_names_keep_distinct_arena_allocation_order():
    a, b = tensor("same", (1,), "int32", align=64), tensor("same", (1,), "int32", align=64)
    result = apply(
        "MergeUBAllocations", [allocate(a), allocate(b), annotation({"tl.buffer_alias_map": {a: [], b: []}})], arguments=(16, True)
    )
    bindings = definitions(result)
    assert address(a, bindings)[1] == 0 and address(b, bindings)[1] == 64


def test_noncompact_allocation_requires_flattening_before_arena_sizing():
    a = tensor("a", (2, 3), "int32", layout=TileLayout(S[(2, 3) : (5, 1)]))
    b = tensor("b", (3,), "int32")
    with pytest.raises(Exception, match="flattened, compact allocation"):
        apply("MergeUBAllocations", [allocate(a), allocate(b), annotation({"tl.buffer_alias_map": {a: [], b: []}})], arguments=(16, False))


def test_arena_rejects_hoisting_a_mutable_load_used_as_extent():
    source = tensor("shape", (1,), "int32", "global")
    extent = ir.Var("extent", "int32")
    a, b = tensor("a", (extent,), "int32"), tensor("b", (2,), "int32")
    with pytest.raises(Exception, match="cannot hoist a memory read or trapping expression"):
        apply(
            "MergeUBAllocations",
            [ir.Bind(extent, ir.TensorLoad(source, [0])), allocate(a), allocate(b), annotation({"tl.buffer_alias_map": {a: [], b: []}})],
            [source],
            arguments=(16, False),
        )


@pytest.mark.parametrize("align", [0, 3, 1 << 28])
def test_invalid_arena_alignment_is_rejected(align):
    a, b = tensor("a"), tensor("b")
    with pytest.raises(Exception, match="power of two"):
        apply(
            "MergeUBAllocations", [allocate(a), allocate(b), annotation({"tl.buffer_alias_map": {a: [], b: []}})], arguments=(align, False)
        )


def test_repeated_inference_keeps_one_canonical_contract_without_empty_growth():
    a, b, gm = tensor("a"), tensor("b"), tensor("gm", scope="global")
    body = [allocate(a), allocate(b), store(a, gm), *flags("MTE3_MTE2"), load(gm, b)]
    first = apply("InferBufferAliases", body, [gm])
    second = apply("InferBufferAliases", first.body, [gm])
    ir.assert_structural_equal(first, second)
    assert len(contract(second)) == 2


def test_repeated_eligibility_preserves_one_native_loop_carrier():
    ub, out = tensor("ub", (1,), "int32"), tensor("out", (4,), "int32", "global")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        [unit(ir.TensorStore(ub, [0], i)), unit(ir.TensorStore(out, [i], ir.TensorLoad(ub, [0])))],
    )
    first = apply("AnnotateMultiBufferEligible", [scheduled([unit(loop)], [ub])], [out])
    second = apply("AnnotateMultiBufferEligible", first.body, [out])
    ir.assert_structural_equal(first, second)
    assert len(loop_claims(objects(second, ir.For)[0])) == 1


@pytest.mark.parametrize("placement,expected", [("inner", "inner"), ("outer", "outer"), ("hoisted", None)])
def test_l0_data_and_scale_choose_one_complete_owner(placement, expected):
    data = tensor("data", (1,), "int32", "shared.l0a")
    scale = tensor("scale", (1,), "int32", "shared.l0a.sf")
    out = tensor("out", (4,), "int32", "global")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    write_scale = unit(ir.TensorStore(scale, [0], integer(2)), core=2)
    inner_body = [
        unit(ir.TensorStore(data, [0], inner), core=2),
        unit(ir.TensorStore(out, [inner], ir.TensorLoad(data, [0]) + ir.TensorLoad(scale, [0])), core=2),
    ]
    if placement == "inner":
        inner_body.insert(0, write_scale)
    outer_body = [unit(ir.For(inner, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, inner_body))]
    if placement == "outer":
        outer_body.insert(0, write_scale)
    body = [unit(ir.For(outer, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, outer_body))]
    if placement == "hoisted":
        body.insert(0, write_scale)
    result = apply("AnnotateMultiBufferEligible", [scheduled(body, [data, scale], {"tl.l0_sf_bindings": {scale: data}})], [out])
    for storage in (data, scale):
        owners = [str(loop.var.name) for loop in objects(result, ir.For) if any(var.same_as(storage) for var in loop_claims(loop))]
        assert owners == ([expected] if expected else [])


@pytest.mark.parametrize("conflict", [False, True])
def test_l0_scale_version_override_covers_the_allocation_group(conflict):
    data = tensor("data", (1,), "int32", "shared.l0a")
    scale = tensor("scale", (1,), "int32", "shared.l0a.sf")
    versions = {scale: 2, **({data: 3} if conflict else {})}
    root = scheduled([], [data, scale], {"tl.l0_sf_bindings": {scale: data}, "tl.buffer_versions_map": versions})
    if conflict:
        with pytest.raises(Exception, match="Conflicting.*bound L0 data/SF group"):
            apply("AnnotateMultiBufferEligible", [root])
    else:
        result = apply("AnnotateMultiBufferEligible", [root])
        # Expanded group members may be added in a separate annotation carrier
        # after their definition. The effective policy must cover each member
        # exactly once, independent of how those carriers are grouped.
        entries = [
            entry for fields in metadata(result) if "tl.buffer_versions_map" in fields for entry in fields["tl.buffer_versions_map"].items()
        ]
        policy = dict(entries)
        assert len(entries) == len(policy) == 2
        assert int(policy[data]) == int(policy[scale]) == 2


@pytest.mark.parametrize("size", [3, 5, 9])
def test_dynamic_version_shape_executes_on_cpu(size, tmp_path):
    n = ir.Var("n", "int32")
    storage = tensor("storage", (2, n), "int32")
    out = tensor("out", (2, n), "int32", "global")
    version, index = ir.Var("version", "int32"), ir.Var("index", "int32")
    inner = ir.For(
        index,
        integer(0),
        n,
        integer(1),
        ir.ForKind.DEFAULT,
        [
            ir.TensorStore(storage, [version, index], version * 100 + index),
            ir.TensorStore(out, [version, index], ir.TensorLoad(storage, [version, index])),
        ],
    )
    loop = ir.For(version, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [inner])
    result = apply("RewriteAscendBufferVersionLayout", [allocate(storage), annotation({"tl.buffer_version": {storage: 2}}), loop], [n, out])
    function = compile_cpu(result, tmp_path)
    array = tvm_tirx.runtime.tensor(np.full((2, size), -1, dtype="int32"))
    function(size, array)
    np.testing.assert_array_equal(array.numpy(), np.arange(size, dtype="int32")[None, :] + np.array([0, 100], dtype="int32")[:, None])
