"""Allocation markers for explicit VMI data slots.

The frontend records ``tl.vmi.alloc_kind = {slot.data: "data"}`` on the
enclosing block.  ``NormalizePTOAllocMarkers`` converts those entries into
scoped ``AttrStmt`` carriers on the block that actually owns the allocation,
and early explicit unroll remaps each carrier's data Var with the clone.
``LowerOpaqueBlock`` preserves the carriers and the data Var, then
``ImportPTOAllocMarkers`` moves the kind onto the final ``AllocBuffer``
annotation consumed by code generation.

These are host-only IR tests: they cover Var identity, not numeric execution.
Compilation and device numerics live under ``testing/ascend/language/pto/``.
"""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import transform as ascend
from tvm import tirx
from tvm.target import Target
from testing.ascend._ir import nodes

_MARKER = "tl.vmi.alloc_kind"
_DATA = "data"
_DATA_ARRAY = "data_array"
_PTO_TARGET = Target({"kind": "ascend", "keys": ["pto", "ascend"]})


def _slot(name="slot", dtype="int32", lanes=1, scope="local.var"):
    element = dtype if lanes == 1 else f"{dtype}x{lanes}"
    return tirx.decl_buffer((1,), element, name=name, scope=scope)


def _array(name="regs", shape=(2,), dtype="float32x64", scope="local"):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _marker(entries):
    return {_MARKER: dict(entries)}


def _carrier(data, kind=_DATA, body=None):
    if body is None:
        body = tirx.Evaluate(0)
    return tirx.AttrStmt(data, _MARKER, kind, body)


def _scope_attr(body):
    return tirx.AttrStmt("simdvf", "tl.simdvf_scope", tirx.IntImm("int32", 1), body)


def _marker_blocks(ir):
    return [block for block in nodes(ir, tirx.SBlock) if _MARKER in block.annotations]


def _mapping(block):
    return block.annotations[_MARKER]


def _carriers(ir):
    return [node for node in nodes(ir, tirx.AttrStmt) if node.attr_key == _MARKER]


def _scope(ir):
    scopes = [node for node in nodes(ir, tirx.AttrStmt) if node.attr_key == "tl.simdvf_scope"]
    assert len(scopes) == 1, scopes
    return scopes[0]


def _seq_statements(stmt):
    if isinstance(stmt, tirx.SeqStmt):
        statements = []
        for child in stmt.seq:
            statements.extend(_seq_statements(child))
        return statements
    return [stmt]


def _module(body):
    return tvm.IRModule({"main": tirx.PrimFunc([], body)})


def _bind_module(body, target=_PTO_TARGET):
    mod = _module(body)
    if target is not None:
        mod = tirx.transform.BindTarget(target)(mod)
    return mod


def _normalize_module(body, target=_PTO_TARGET):
    return ascend.NormalizePTOAllocMarkers()(_bind_module(body, target))


def _normalize(body, target=_PTO_TARGET):
    return _normalize_module(body, target)["main"].body


def _unroll(body):
    # Marker-carrying functions bind the PTO target and turn block maps into
    # AttrStmt carriers before the early unroll clones them.
    mod = _normalize_module(body)
    return ascend.UnrollLoopSkipVF()(mod)["main"].body


def _lower_module(body):
    return tilelang.transform.LowerOpaqueBlock()(_normalize_module(body))


def _lower(body):
    return _lower_module(body)["main"].body


def _imported(body, target=_PTO_TARGET):
    # BindTarget -> Normalize -> LowerOpaqueBlock -> Import.
    mod = _lower_module(body)
    if target is not None:
        mod = tirx.transform.BindTarget(target)(mod)
    return ascend.ImportPTOAllocMarkers()(mod)["main"].body


def _import_raw(body, target=_PTO_TARGET):
    mod = _bind_module(body, target)
    return ascend.ImportPTOAllocMarkers()(mod)["main"].body


def _loop(body_fn, *, extent=2, kind=tirx.ForKind.UNROLLED, explicit=True):
    var = tirx.Var("i", "int32")
    annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    return tirx.For(var, 0, extent, kind, body_fn(var), annotations=annotations)


def _vf(body, alloc_buffers=(), *, scoped=True):
    if scoped:
        body = _scope_attr(body)
    return tirx.SBlock([], [], [], "SIMD_VF", body, alloc_buffers=list(alloc_buffers))


def _slot_uses(ir):
    stores = [node for node in nodes(ir, tirx.BufferStore) if node.buffer.scope() == "local.var"]
    loads = [node for node in nodes(ir, tirx.BufferLoad) if node.buffer.scope() == "local.var"]
    return stores, loads


def _owner_of(ir, data):
    for block in nodes(ir, tirx.SBlock):
        if any(buffer.data.same_as(data) for buffer in block.alloc_buffers):
            return block
    raise AssertionError(f"no block owns the data Var {data}")


def _assert_carrier_inside_scope(block, data, kind):
    scope = _scope(block)
    carriers = [node for node in nodes(scope.body, tirx.AttrStmt) if node.attr_key == _MARKER and node.node.same_as(data)]
    assert len(carriers) == 1, carriers
    (carrier,) = carriers
    assert isinstance(carrier.value, tirx.StringImm)
    assert str(carrier.value.value) == kind


# ---------------------------------------------------------------------------
# Frontend registration and scope
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_frontend_registers_each_explicit_slot_once():
    import tilelang.ascend.language as T

    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((256,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work[0:128])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot0 = T.vmi.alloc_var("int32", size=1)
                slot1 = T.vmi.alloc_var("int32", size=4)
                slot0 = T.vmi.vcmax(x, mask)
                slot1 = T.vmi.vload(work[0], size=4)
                T.vmi.vstore(T.vmi.vbrc(slot0, size=128, group=1), out[0], mask)
                T.vmi.vstore(T.vmi.vbrc(slot1, size=128, group=4), out[0], mask)
            T.copy(out, C)

    blocks = _marker_blocks(main.body)
    assert len(blocks) == 1, "all frontend slots merge into one block map"
    mapping = _mapping(blocks[0])
    assert len(mapping) == 2
    assert all(str(value) == _DATA for value in mapping.values())

    (vf,) = [block for block in nodes(main.body, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    assert len(vf.alloc_buffers) == 2
    keys = list(mapping.keys())
    for key in keys:
        assert any(key.same_as(buffer.data) for buffer in vf.alloc_buffers)
    assert not keys[0].same_as(keys[1])


@pytest.mark.pto
def test_frontend_registers_array_and_slot_kind_per_allocation():
    # A VMI data array and a pure SIMD register array can have identical
    # shape, dtype, and scope.  Only the VMI allocation is marked, and its
    # kind distinguishes it from a lane-1 slot.
    import tilelang.ascend.language as T

    @T.prim_func
    def vmi_mix(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                regs_a = T.vmi.alloc_local((2,), T.vmi.vreg(64, "float32"))
                regs_b = T.vmi.alloc_local((3,), T.vmi.vreg(64, "float32"))
                slot = T.vmi.alloc_var("float32", size=1)
                slot = T.vmi.vcmax(x, mask)
                regs_a[0] = T.vmi.vload(work[0], size=64)
                regs_b[1] = T.vmi.vload(work[64], size=64)
                T.vmi.vstore(T.vmi.vbrc(slot, size=128, group=1), out[0], mask)
            T.copy(out, C)

    @T.prim_func
    def simd_array(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, work)
            with T.SimdVF():
                regs = T.simd.alloc_local((2,), "float32")
                regs[0] = T.simd.vld(work[0])
                regs[1] = T.simd.vld(work[64])
                T.simd.vsts(out[0], regs[0])
            T.copy(out, C)

    blocks = _marker_blocks(vmi_mix.body)
    assert len(blocks) == 1, "the VMI block map is recorded once"
    mapping = _mapping(blocks[0])
    (vf,) = [block for block in nodes(vmi_mix.body, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    allocations = list(vf.alloc_buffers)
    assert len(allocations) == 3
    assert len(mapping) == len(allocations), "one entry per marked VMI allocation"

    def kind_for(data):
        matches = [str(value) for key, value in mapping.items() if key.same_as(data)]
        assert len(matches) == 1, matches
        return matches[0]

    for allocation in allocations:
        assert any(key.same_as(allocation.data) for key in mapping)
        expected = "data" if allocation.scope() == "local.var" else "data_array"
        assert kind_for(allocation.data) == expected

    keys = list(mapping.keys())
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            assert not left.same_as(right)

    # The pure SIMD allocation has the same shape/dtype/scope as the VMI data
    # array but never receives a marker.
    assert not _marker_blocks(simd_array.body)
    (simd_vf,) = [block for block in nodes(simd_array.body, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    (simd_regs,) = simd_vf.alloc_buffers
    (vmi_regs,) = [buffer for buffer in allocations if buffer.name == "regs_a"]
    assert tuple(int(dim) for dim in simd_regs.shape) == tuple(int(dim) for dim in vmi_regs.shape)
    assert str(simd_regs.dtype) == str(vmi_regs.dtype)
    assert simd_regs.scope() == vmi_regs.scope() == "local"


@pytest.mark.pto
def test_frontend_rejects_mask_and_handle_register_arrays():
    # Only numeric data arrays are supported.  A mask element type would be
    # reclassified as VMI data on read, and a handle is not register data.
    import tilelang.ascend.language as T

    def build(vreg_type):
        @T.prim_func
        def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
            with T.Kernel(1):
                work = T.alloc_shared((128,), "float32")
                out = T.alloc_shared((128,), "float32")
                T.copy(A, work)
                with T.SimdVF():
                    T.vmi.alloc_local((2,), vreg_type)
                T.copy(out, C)

        return main

    with pytest.raises(TypeError, match="numeric non-bool"):
        build(T.vmi.mask(128))
    with pytest.raises(TypeError, match="numeric non-bool"):
        build(T.vmi.vreg(64, "handle"))


@pytest.mark.pto
def test_frontend_accepts_wide_and_tfloat_array_elements():
    # The element contract is numeric, not a width whitelist: float64 and the
    # canonical tfloat32 vector descriptor are both valid register array
    # elements.  This only covers allocation; neither width gains an
    # instruction-level promise here.
    import tilelang.ascend.language as T

    def build(vreg_type):
        @T.prim_func
        def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
            with T.Kernel(1):
                work = T.alloc_shared((128,), "float32")
                out = T.alloc_shared((128,), "float32")
                T.copy(A, work)
                with T.SimdVF():
                    T.vmi.alloc_local((2,), vreg_type)
                T.copy(out, C)

        return main

    cases = {
        "float64x64": T.vmi.vreg(64, "float64"),
        "custom[tfloat32]x64": T.vmi.vreg(64, T.tfloat32),
    }
    for expected, vreg_type in cases.items():
        func = build(vreg_type)
        (block,) = _marker_blocks(func.body)
        (vf,) = [node for node in nodes(func.body, tirx.SBlock) if node.name_hint == "SIMD_VF"]
        (allocation,) = vf.alloc_buffers
        assert str(allocation.dtype) == expected
        mapping = _mapping(block)
        assert len(mapping) == 1
        (key,) = mapping.keys()
        assert key.same_as(allocation.data)
        assert str(mapping[key]) == _DATA_ARRAY


@pytest.mark.parametrize("api", ["alloc_local", "alloc_var"])
@pytest.mark.pto
def test_frontend_vmi_allocations_require_simdvf_scope(api):
    import tilelang.ascend.language as T

    def build():
        @T.prim_func
        def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
            with T.Kernel(1):
                if api == "alloc_local":
                    T.vmi.alloc_local((1,), T.vmi.vreg(64, "float32"))
                else:
                    T.vmi.alloc_var("float32", size=1)
                T.copy(A, C)

        return main

    with pytest.raises(RuntimeError, match=r"operations require a `T\.SimdVF\(\)` scope"):
        build()


@pytest.mark.pto
def test_frontend_ordinary_alloc_var_keeps_its_unrestricted_scope():
    # The VMI APIs are SIMD_VF-only; the ordinary allocation API is not.
    import tilelang.ascend.language as T

    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            slot = T.alloc_var("int32")
            slot = 1  # noqa: F841 - the assignment itself is the behavior under test.
            T.copy(A, C)

    assert not _marker_blocks(main.body)


# ---------------------------------------------------------------------------
# Ownership normalization emits scoped AttrStmt carriers
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_normalize_moves_marker_to_allocation_owner():
    slot_a = _slot(name="duplicate")
    slot_b = _slot(name="duplicate")
    slot_c = _slot(name="nested_owner")
    slot_d = _slot(name="second")

    nested = tirx.SBlock(
        [],
        [],
        [],
        "nested_owner_block",
        tirx.BufferStore(slot_c, tirx.IntImm("int32", 3), [0]),
        alloc_buffers=[slot_c],
        annotations={"tl.owner_probe": "owner"},
    )
    vf_one = _vf(
        tirx.BufferStore(slot_a, tirx.IntImm("int32", 1), [0]),
        alloc_buffers=[slot_a],
    )
    vf_two = _vf(
        tirx.SeqStmt(
            [
                tirx.BufferStore(slot_b, tirx.IntImm("int32", 2), [0]),
                tirx.BufferStore(slot_d, tirx.IntImm("int32", 4), [0]),
                tirx.SBlockRealize([], True, nested),
            ]
        ),
        alloc_buffers=[slot_b, slot_d],
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.SeqStmt([vf_one, vf_two]),
        annotations={
            _MARKER: {slot_a.data: _DATA, slot_c.data: _DATA, slot_d.data: _DATA},
            "tl.owner_probe": "kept",
        },
    )
    after = _normalize(root)

    # The outer map is consumed; unrelated annotations survive on the block.
    (root_after,) = [b for b in nodes(after, tirx.SBlock) if b.name_hint == "tilelang_root"]
    assert _MARKER not in root_after.annotations
    assert root_after.annotations["tl.owner_probe"] == "kept"
    assert not _marker_blocks(after)

    # Each owner carries only its own marked allocation, with the original
    # data Var and the allocation still in place.  The marker sits inside the
    # existing outermost SIMD_VF scope statement.
    owner_a = _owner_of(after, slot_a.data)
    assert owner_a.name_hint == "SIMD_VF"
    (allocation_a,) = owner_a.alloc_buffers
    assert allocation_a.data.same_as(slot_a.data)
    _assert_carrier_inside_scope(owner_a, slot_a.data, _DATA)

    # The unmarked same-name twin in the other VF gains no carrier by name;
    # the multi-slot owner emits only its own marked slot.
    owner_b = _owner_of(after, slot_b.data)
    owner_d = _owner_of(after, slot_d.data)
    assert owner_b.same_as(owner_d)
    assert not any(carrier.node.same_as(slot_b.data) for carrier in _carriers(owner_b))
    _assert_carrier_inside_scope(owner_b, slot_d.data, _DATA)
    (allocation_b,) = [b for b in owner_b.alloc_buffers if b.data.same_as(slot_b.data)]
    assert allocation_b.data.same_as(slot_b.data)

    # A plain non-VF block nested inside the VF may own an allocation; its
    # carrier is emitted in that block's body and its annotations survive.
    owner_c = _owner_of(after, slot_c.data)
    assert owner_c.name_hint == "nested_owner_block"
    (allocation_c,) = owner_c.alloc_buffers
    assert allocation_c.data.same_as(slot_c.data)
    assert owner_c.annotations["tl.owner_probe"] == "owner"
    (carrier_c,) = [carrier for carrier in _carriers(owner_c) if carrier.node.same_as(slot_c.data)]
    assert str(carrier_c.value.value) == _DATA
    (realize,) = nodes(after, tirx.SBlockRealize)
    assert realize.block.same_as(owner_c)

    # Ownership metadata only: no allocation is materialized early and every
    # slot use still refers to an original data Var.
    assert not nodes(after, tirx.AllocBuffer)
    stores, _ = _slot_uses(after)
    assert len(stores) == 4
    originals = [slot_a.data, slot_b.data, slot_c.data, slot_d.data]
    for store in stores:
        assert any(store.buffer.data.same_as(data) for data in originals)

    # The carriers are now in the regions normalization emits them into, so a
    # repeated pass must accept them unchanged, including the nested owner.
    rebound = _bind_module(after)
    repeated = ascend.NormalizePTOAllocMarkers()(rebound)
    assert repeated["main"].body.same_as(rebound["main"].body)


@pytest.mark.pto
def test_normalize_emits_carriers_in_allocation_order():
    first = _slot(name="first")
    second = _slot(name="second")
    vf = _vf(
        tirx.SeqStmt(
            [
                tirx.BufferStore(first, tirx.IntImm("int32", 1), [0]),
                tirx.BufferStore(second, tirx.IntImm("int32", 2), [0]),
            ]
        ),
        alloc_buffers=[first, second],
    )
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({first.data: _DATA, second.data: _DATA}))
    after = _normalize(root)

    (owner,) = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    scope = _scope(owner)
    assert isinstance(scope.body, tirx.AttrStmt)
    assert scope.body.attr_key == _MARKER
    assert scope.body.node.same_as(first.data)
    assert isinstance(scope.body.body, tirx.AttrStmt)
    assert scope.body.body.attr_key == _MARKER
    assert scope.body.body.node.same_as(second.data)


@pytest.mark.pto
def test_normalize_array_marker_is_idempotent():
    regs = _array()
    vf = _vf(
        tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
        alloc_buffers=[regs],
    )
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({regs.data: _DATA_ARRAY}))
    once = _normalize_module(root)
    twice = ascend.NormalizePTOAllocMarkers()(once)

    # The carrier is already on the owner, so a second run leaves the function
    # untouched instead of adding another marker authority.
    assert once["main"].body.same_as(twice["main"].body)

    (owner,) = [block for block in nodes(once["main"].body, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    assert not _marker_blocks(once["main"].body)
    _assert_carrier_inside_scope(owner, regs.data, _DATA_ARRAY)
    assert len(_carriers(once["main"].body)) == 1


@pytest.mark.pto
def test_normalize_keeps_existing_allocation_annotation_without_carrier():
    slot = _slot()
    vf = _vf(
        tirx.SeqStmt(
            [
                tirx.AllocBuffer(slot, annotations={_MARKER: _DATA}),
                tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
            ]
        )
    )
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({slot.data: _DATA}))
    after = _normalize(root)

    assert not _marker_blocks(after)
    assert not _carriers(after)
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert str(allocation.annotations[_MARKER]) == _DATA
    assert allocation.buffer.data.same_as(slot.data)


@pytest.mark.pto
def test_normalize_attaches_kind_to_statement_allocation():
    slot = _slot()
    vf = _vf(
        tirx.SeqStmt(
            [
                tirx.AllocBuffer(slot),
                tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
            ]
        )
    )
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({slot.data: _DATA}))
    after = _normalize(root)

    # A statement allocation is already in final form: its kind is attached
    # directly instead of wrapping it in a second carrier.
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert str(allocation.annotations[_MARKER]) == _DATA
    assert allocation.buffer.data.same_as(slot.data)
    assert not _carriers(after)
    stores, _ = _slot_uses(after)
    assert len(stores) == 1


@pytest.mark.pto
def test_normalize_drops_residual_map_entry_but_emits_the_real_carrier():
    slot = _slot()
    stale = _slot(name="stale")
    vf = _vf(
        tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
        alloc_buffers=[slot],
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        vf,
        annotations=_marker({slot.data: _DATA, stale.data: _DATA}),
    )
    after = _normalize(root)

    # A map entry without an allocation is stale metadata: it is dropped with
    # the consumed block map and never matched to the allocation by position or
    # name.  The real entry still produces exactly one carrier.
    assert not _marker_blocks(after)
    assert len(_carriers(after)) == 1
    (carrier,) = _carriers(after)
    assert carrier.node.same_as(slot.data)
    assert not any(carrier.node.same_as(stale.data) for carrier in _carriers(after))
    _assert_carrier_inside_scope(_owner_of(after, slot.data), slot.data, _DATA)


# ---------------------------------------------------------------------------
# Early explicit unroll clones the carrier with the allocation Var
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_unroll_copies_marker_to_each_cloned_vf_allocation():
    out = tirx.decl_buffer((2,), "int32", name="out")
    slot = _slot()

    def body(i):
        return _vf(
            tirx.SeqStmt(
                [
                    tirx.BufferStore(slot, i, [0]),
                    tirx.BufferStore(out, tirx.BufferLoad(slot, [0]), [i]),
                ]
            ),
            alloc_buffers=[slot],
        )

    root = tirx.SBlock([], [], [], "", _loop(body), annotations=_marker({slot.data: _DATA}))
    after = _unroll(root)

    vfs = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    assert len(vfs) == 2
    assert not _marker_blocks(after)
    data_vars = []
    out_indices = []
    for vf in vfs:
        (allocation,) = vf.alloc_buffers
        _assert_carrier_inside_scope(vf, allocation.data, _DATA)
        assert len(_carriers(vf)) == 1
        # Each clone reads and writes its own slot only: one store of the
        # induction value and one load feeding the distinct output store.
        stores, loads = _slot_uses(vf)
        assert len(stores) == 1
        assert stores[0].buffer.data.same_as(allocation.data)
        assert len(loads) == 1
        assert loads[0].buffer.data.same_as(allocation.data)
        (out_store,) = [node for node in nodes(vf, tirx.BufferStore) if node.buffer.data.same_as(out.data)]
        assert isinstance(out_store.value, tirx.BufferLoad)
        assert out_store.value.buffer.data.same_as(allocation.data)
        assert int(out_store.value.indices[0]) == 0
        out_indices.append(int(out_store.indices[0]))
        data_vars.append(allocation.data)
    assert not data_vars[0].same_as(data_vars[1])
    assert sorted(out_indices) == [0, 1]

    # No surviving carrier names the original data Var: each clone references
    # only its own fresh allocation.
    carrier_nodes = [carrier.node for carrier in _carriers(after)]
    assert len(carrier_nodes) == len(vfs)
    for carrier_data in carrier_nodes:
        assert not carrier_data.same_as(slot.data)
    assert all(any(carrier_data.same_as(data) for data in data_vars) for carrier_data in carrier_nodes)


@pytest.mark.pto
def test_unroll_clones_array_allocation_and_marker():
    out = tirx.decl_buffer((2,), "float32x64", name="out")
    regs = _array(name="regs")
    # Same name, different data Var, and deliberately unmarked: the clone must
    # follow identities, never names or allocation order.
    twin = _array(name="regs")

    def body(i):
        marked = _vf(
            tirx.SeqStmt(
                [
                    tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
                    tirx.BufferStore(out, tirx.BufferLoad(regs, [1]), [i]),
                ]
            ),
            alloc_buffers=[regs],
        )
        unmarked = _vf(
            tirx.BufferStore(twin, tirx.BufferLoad(twin, [0]), [0]),
            alloc_buffers=[twin],
        )
        return tirx.SeqStmt([marked, unmarked])

    root = tirx.SBlock([], [], [], "", _loop(body), annotations=_marker({regs.data: _DATA_ARRAY}))
    after = _unroll(root)

    vfs = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    assert len(vfs) == 4, "both the marked array and its unmarked twin are cloned"
    marked_vfs = [block for block in vfs if _carriers(block)]
    assert len(marked_vfs) == 2

    data_vars = []
    out_indices = []
    for block in marked_vfs:
        (allocation,) = block.alloc_buffers
        _assert_carrier_inside_scope(block, allocation.data, _DATA_ARRAY)
        # Count the register accesses inside this clone block only: a single
        # store into the clone array plus the two loads that feed that store
        # and the output.  Selecting by data Var over the whole module would
        # accept cross-clone uses.
        stores = [node for node in nodes(block, tirx.BufferStore) if node.buffer.scope() == "local"]
        loads = [node for node in nodes(block, tirx.BufferLoad) if node.buffer.scope() == "local"]
        assert len(stores) == 1
        assert stores[0].buffer.data.same_as(allocation.data)
        assert len(loads) == 2
        assert all(load.buffer.data.same_as(allocation.data) for load in loads)
        (out_store,) = [node for node in nodes(block, tirx.BufferStore) if node.buffer.data.same_as(out.data)]
        assert isinstance(out_store.value, tirx.BufferLoad)
        assert out_store.value.buffer.data.same_as(allocation.data)
        assert int(out_store.value.indices[0]) == 1
        out_indices.append(int(out_store.indices[0]))
        data_vars.append(allocation.data)
    assert not data_vars[0].same_as(data_vars[1])
    assert sorted(out_indices) == [0, 1]

    # The unmarked twin allocations never receive a carrier, and no surviving
    # carrier names either original data Var.
    twin_vars = []
    for block in vfs:
        if _carriers(block):
            continue
        (twin,) = block.alloc_buffers
        twin_vars.append(twin.data)
        assert all(node.buffer.data.same_as(twin.data) for node in nodes(block, tirx.BufferStore) if node.buffer.scope() == "local")
        assert all(node.buffer.data.same_as(twin.data) for node in nodes(block, tirx.BufferLoad) if node.buffer.scope() == "local")
    assert len(twin_vars) == 2
    for carrier in _carriers(after):
        assert not carrier.node.same_as(regs.data)
        assert not any(carrier.node.same_as(data) for data in twin_vars)


@pytest.mark.pto
def test_unroll_keeps_original_data_var_when_no_clone_rebuilds_the_allocation():
    # The carrier is inside the cloned loop, but its allocation lives on the
    # enclosing block, so the freshener has no replacement Var for it.  The
    # carrier must then keep the original identity rather than become dangling.
    outer = _slot(name="outer")
    loop = _loop(lambda i: _carrier(outer.data))
    root = tirx.SBlock([], [], [], "tilelang_root", loop, alloc_buffers=[outer])
    after = ascend.UnrollLoopSkipVF()(_module(root))["main"].body

    carriers = _carriers(after)
    assert len(carriers) == 2
    assert all(carrier.node.same_as(outer.data) for carrier in carriers)
    assert not nodes(after, tirx.For)


@pytest.mark.pto
def test_late_unroll_clones_marked_array_allocation_and_marker():
    # LowerOpaqueBlock hands allocations to code generation as AllocBuffer
    # statements.  If one is still inside an explicitly unrolled body at the
    # late UnrollLoop pass, that pass clones the allocation.  Every clone must
    # own a fresh data Var, keep the data_array string marker, and read/write
    # only its own registers.
    out = tirx.decl_buffer((2,), "int32x64", name="out", scope="local")
    regs = _array(dtype="int32x64")
    i = tirx.Var("i", "int32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(regs, annotations={_MARKER: _DATA_ARRAY}),
            tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
            tirx.BufferStore(out, tirx.BufferLoad(regs, [1]), [i]),
        ]
    )
    loop = tirx.For(
        i,
        0,
        2,
        tirx.ForKind.UNROLLED,
        body,
        annotations={"pragma_unroll_explicit": True},
    )
    after = tilelang.transform.UnrollLoop()(_module(loop))["main"].body

    statements = _seq_statements(after)
    assert not any(isinstance(stmt, tirx.For) for stmt in statements)
    segments = []
    for stmt in statements:
        if isinstance(stmt, tirx.AllocBuffer):
            segments.append((stmt, []))
        else:
            assert segments, "a cloned statement appeared before its allocation"
            segments[-1][1].append(stmt)
    assert len(segments) == 2

    data_vars = []
    out_indices = []
    for allocation, clone_body in segments:
        assert str(allocation.annotations[_MARKER]) == _DATA_ARRAY
        data = allocation.buffer.data
        assert not data.same_as(regs.data)
        stores = [node for stmt in clone_body for node in nodes(stmt, tirx.BufferStore)]
        loads = [node for stmt in clone_body for node in nodes(stmt, tirx.BufferLoad)]
        assert len([node for node in stores if node.buffer.data.same_as(data)]) == 1
        assert len(loads) == 2
        assert all(node.buffer.data.same_as(data) for node in loads)
        (out_store,) = [node for node in stores if node.buffer.data.same_as(out.data)]
        assert isinstance(out_store.value, tirx.BufferLoad)
        assert out_store.value.buffer.data.same_as(data)
        assert int(out_store.value.indices[0]) == 1
        out_indices.append(int(out_store.indices[0]))
        data_vars.append(data)
    assert not data_vars[0].same_as(data_vars[1])
    assert sorted(out_indices) == [0, 1]


@pytest.mark.pto
def test_normalize_rejects_allocation_outside_simd_vf():
    slot = _slot()
    outside = tirx.SBlock(
        [],
        [],
        [],
        "plain",
        tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0]),
        alloc_buffers=[slot],
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.SeqStmt([outside, _vf(tirx.Evaluate(0))]),
        annotations=_marker({slot.data: _DATA}),
    )
    with pytest.raises(tvm.error.InternalError, match="SIMD_VF"):
        _normalize(root)


@pytest.mark.pto
def test_normalize_rejects_vf_owner_without_scope_statement():
    # A SIMD_VF owner must already have the scope statement as its outermost
    # body; emitting carriers there would otherwise move them outside it.
    slot = _slot()
    owner = _vf(
        tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0]),
        alloc_buffers=[slot],
        scoped=False,
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        owner,
        annotations=_marker({slot.data: _DATA}),
    )
    with pytest.raises(tvm.error.InternalError, match="outermost statement is not"):
        _normalize(root)


@pytest.mark.parametrize("placement", ["outside-vf", "other-vf"])
@pytest.mark.pto
def test_normalize_rejects_existing_carrier_outside_owning_vf(placement):
    # A carrier for a marked allocation is only an existing marker when it is
    # in the region normalization would emit it into.  A carrier at the root,
    # or in a different SIMD_VF, must not suppress emission for the real owner.
    slot = _slot()
    owner = _vf(
        tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
        alloc_buffers=[slot],
    )
    if placement == "outside-vf":
        body = tirx.SeqStmt([_carrier(slot.data), owner])
    else:
        body = tirx.SeqStmt([owner, _vf(_carrier(slot.data))])
    root = tirx.SBlock([], [], [], "tilelang_root", body)

    with pytest.raises(tvm.error.InternalError, match="owning marker region"):
        _normalize(root)


# ---------------------------------------------------------------------------
# LowerOpaqueBlock preserves carriers; Import attaches final annotations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("realized", [False, True], ids=["direct", "realized"])
@pytest.mark.pto
def test_lower_opaque_preserves_carrier_and_import_transfers(realized):
    slot = _slot()
    body = tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0])
    owner = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[slot],
        annotations=_marker({slot.data: _DATA}),
    )
    wrapped = tirx.SBlockRealize([], True, owner) if realized else owner
    root = _vf(wrapped)

    lowered = _lower(root)
    (allocation,) = nodes(lowered, tirx.AllocBuffer)
    assert allocation.buffer.data.same_as(slot.data)
    assert _MARKER not in allocation.annotations
    assert not _marker_blocks(lowered)
    (carrier,) = [node for node in _carriers(lowered) if node.node.same_as(slot.data)]
    assert str(carrier.value.value) == _DATA

    imported = _imported(root)
    (allocation,) = nodes(imported, tirx.AllocBuffer)
    assert allocation.buffer.data.same_as(slot.data)
    assert str(allocation.annotations[_MARKER]) == _DATA
    assert not _carriers(imported)
    assert not _marker_blocks(imported)
    stores, _ = _slot_uses(imported)
    assert len(stores) == 1


@pytest.mark.pto
def test_lower_opaque_materializes_allocation_before_scope_marker():
    # The final allocation may be materialized lexically before the
    # tl.simdvf_scope AttrStmt.  Import must match it by data Var identity, not
    # by lexical containment inside the scope.
    slot = _slot()
    vf = _vf(tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0]), alloc_buffers=[slot])
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({slot.data: _DATA}))

    lowered = _lower(root)
    (vf_lowered,) = [block for block in nodes(lowered, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    statements = _seq_statements(vf_lowered.body)
    allocation_index = next(index for index, stmt in enumerate(statements) if isinstance(stmt, tirx.AllocBuffer))
    scope_index = next(
        index for index, stmt in enumerate(statements) if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "tl.simdvf_scope"
    )
    assert allocation_index < scope_index
    assert vf_lowered.alloc_buffers == []

    imported = _imported(root)
    (allocation,) = nodes(imported, tirx.AllocBuffer)
    assert str(allocation.annotations[_MARKER]) == _DATA
    assert not _carriers(imported)


@pytest.mark.parametrize("realized", [False, True], ids=["direct", "realized"])
@pytest.mark.pto
def test_lower_opaque_transfers_array_kind_and_leaves_unmarked_alloc(realized):
    regs = _array(shape=(2, 3))
    slot = _slot()
    plain = _array(name="plain", shape=(2, 3))
    body = tirx.SeqStmt(
        [
            tirx.BufferStore(regs, tirx.BufferLoad(regs, [1, 2]), [1, 2]),
            tirx.BufferStore(plain, tirx.BufferLoad(plain, [0, 0]), [0, 0]),
            tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0]),
        ]
    )
    owner = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[regs, slot, plain],
        annotations=_marker({regs.data: _DATA_ARRAY, slot.data: _DATA}),
    )
    wrapped = tirx.SBlockRealize([], True, owner) if realized else owner
    after = _imported(_vf(wrapped))

    allocations = {node.buffer.name: node for node in nodes(after, tirx.AllocBuffer)}
    assert str(allocations["regs"].annotations[_MARKER]) == _DATA_ARRAY
    assert str(allocations["slot"].annotations[_MARKER]) == _DATA
    assert _MARKER not in allocations["plain"].annotations
    # The carrier is consumed, not duplicated next to the allocations.
    assert not _carriers(after)
    assert not _marker_blocks(after)


@pytest.mark.parametrize("dtype", ["float64x64", "custom[tfloat32]x64"])
@pytest.mark.pto
def test_lower_opaque_accepts_wide_and_tfloat_array_markers(dtype):
    # The allocation validator accepts every numeric width, including float64
    # and tfloat32, and excludes only bool/handle.  This is the marker
    # contract, not instruction support for those element widths.
    regs = _array(dtype=dtype)
    owner = tirx.SBlock(
        [],
        [],
        [],
        "",
        tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
        alloc_buffers=[regs],
        annotations=_marker({regs.data: _DATA_ARRAY}),
    )
    after = _imported(_vf(tirx.SBlockRealize([], True, owner)))
    (allocation,) = [node for node in nodes(after, tirx.AllocBuffer)]
    assert str(allocation.annotations[_MARKER]) == _DATA_ARRAY
    assert str(allocation.buffer.dtype) == dtype


@pytest.mark.parametrize(
    "make_marker,message",
    [
        (lambda slot: {_MARKER: _DATA}, "to be a map"),
        (
            lambda slot: {_MARKER: {"not_a_var": _DATA}},
            "map keys must be allocation data Vars",
        ),
        (lambda slot: {_MARKER: {slot.data: 7}}, "map values must be strings"),
    ],
    ids=["not-a-map", "non-var-key", "non-string-value"],
)
@pytest.mark.pto
def test_normalize_rejects_malformed_annotation_container(make_marker, message):
    slot = _slot()
    block = tirx.SBlock(
        [],
        [],
        [],
        "",
        tirx.Evaluate(0),
        alloc_buffers=[slot],
        annotations=make_marker(slot),
    )
    with pytest.raises(tvm.error.InternalError, match=message):
        _normalize(_vf(tirx.SBlockRealize([], True, block)))


@pytest.mark.pto
def test_normalize_rejects_invalid_allocation_markers():
    def normalized(buffer, kind):
        block = tirx.SBlock(
            [],
            [],
            [],
            "",
            tirx.Evaluate(0),
            alloc_buffers=[buffer],
            annotations={_MARKER: {buffer.data: kind}},
        )
        return _normalize(_vf(tirx.SBlockRealize([], True, block)))

    regs = _array()
    with pytest.raises(tvm.error.InternalError, match="supports only"):
        normalized(regs, "mask_array")
    with pytest.raises(tvm.error.InternalError, match="may only mark a local.var allocation"):
        normalized(regs, _DATA)
    with pytest.raises(tvm.error.InternalError, match="may only mark a local allocation"):
        normalized(_slot(), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="numeric non-bool element type"):
        normalized(_array(name="bool_regs", dtype="boolx64"), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="multi-lane vector element dtype"):
        normalized(tirx.decl_buffer((2,), "float32", name="lane1", scope="local"), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="positive compile-time integer"):
        normalized(_array(name="dynamic", shape=(tirx.Var("n", "int32"),)), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="positive compile-time integer"):
        normalized(_array(name="empty", shape=(0,)), _DATA_ARRAY)


@pytest.mark.parametrize(
    ("make_statement", "message"),
    [
        (lambda slot: _carrier(slot.data), "without an active allocation"),
        (
            lambda slot: tirx.SeqStmt(
                [
                    tirx.AllocBuffer(slot),
                    tirx.AttrStmt(tirx.IntImm("int32", 0), _MARKER, _DATA, tirx.Evaluate(0)),
                ]
            ),
            "carrier node",
        ),
        (
            lambda slot: tirx.SeqStmt(
                [
                    tirx.AllocBuffer(slot),
                    tirx.AttrStmt(slot.data, _MARKER, tirx.IntImm("int32", 1), tirx.Evaluate(0)),
                ]
            ),
            "carrier value",
        ),
    ],
    ids=["orphan", "non-var-node", "non-string-value"],
)
@pytest.mark.pto
def test_import_rejects_malformed_carrier(make_statement, message):
    slot = _slot()
    with pytest.raises(tvm.error.InternalError, match=message):
        _import_raw(make_statement(slot))


@pytest.mark.pto
def test_marker_representations_reject_conflicting_kinds():
    slot = _slot()

    # Carrier vs carrier.
    carrier_pair = tirx.SeqStmt(
        [
            tirx.AllocBuffer(slot),
            _carrier(slot.data, _DATA, _carrier(slot.data, _DATA_ARRAY)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="conflicting"):
        _import_raw(carrier_pair)

    # Final allocation annotation vs carrier.
    annotated = tirx.SeqStmt(
        [
            tirx.AllocBuffer(slot, annotations={_MARKER: _DATA_ARRAY}),
            _carrier(slot.data, _DATA),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="conflicting"):
        _import_raw(annotated)

    # Frontend block map vs carrier.
    vf = _vf(
        tirx.SeqStmt(
            [
                _carrier(slot.data, _DATA),
                tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
            ]
        ),
        alloc_buffers=[slot],
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        vf,
        annotations=_marker({slot.data: _DATA_ARRAY}),
    )
    with pytest.raises(tvm.error.InternalError, match="conflicting"):
        _normalize(root)


@pytest.mark.pto
def test_import_rejects_invalid_carrier_kind_and_allocation_contract():
    slot = _slot()
    invalid_kind = tirx.SeqStmt([tirx.AllocBuffer(slot), _carrier(slot.data, "mask_array")])
    with pytest.raises(tvm.error.InternalError, match="supports only"):
        _import_raw(invalid_kind)

    shared = tirx.decl_buffer((1,), "int32", name="shared_slot", scope="shared")
    invalid_scope = tirx.SeqStmt([tirx.AllocBuffer(shared), _carrier(shared.data)])
    with pytest.raises(tvm.error.InternalError, match="may only mark a local.var allocation"):
        _import_raw(invalid_scope)


@pytest.mark.parametrize("target", [None, Target("cuda")], ids=["no-target", "non-pto"])
@pytest.mark.pto
def test_marker_passes_are_no_ops_without_a_pto_target(target):
    slot = _slot()
    body = tirx.SeqStmt([tirx.AllocBuffer(slot), _carrier(slot.data)])
    mod = _bind_module(body, target)

    normalized = ascend.NormalizePTOAllocMarkers()(mod)
    assert normalized["main"].body.same_as(mod["main"].body)
    imported = ascend.ImportPTOAllocMarkers()(mod)
    assert imported["main"].body.same_as(mod["main"].body)
    assert _carriers(imported)


# ---------------------------------------------------------------------------
# Array markers survive flattening and keep one register index per element
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_flatten_preserves_array_marker_and_index():
    regs = _array(shape=(2, 3))
    body = tirx.BufferStore(regs, tirx.BufferLoad(regs, [0, 0]), [1, 2])
    owner = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[regs],
        annotations=_marker({regs.data: _DATA_ARRAY}),
    )
    lowered = _imported(_vf(tirx.SBlockRealize([], True, owner)))
    (allocation,) = nodes(lowered, tirx.AllocBuffer)
    assert str(allocation.annotations[_MARKER]) == _DATA_ARRAY

    mod = tvm.IRModule({"main": tirx.PrimFunc([], lowered)})
    flat = tilelang.transform.FlattenBuffer()(mod)["main"].body

    (flat_allocation,) = nodes(flat, tirx.AllocBuffer)
    assert flat_allocation.buffer.data.same_as(regs.data)
    assert tuple(int(dim) for dim in flat_allocation.buffer.shape) == (6,)
    assert str(flat_allocation.buffer.dtype) == "float32x64"
    assert str(flat_allocation.annotations[_MARKER]) == _DATA_ARRAY

    (store,) = nodes(flat, tirx.BufferStore)
    assert int(store.indices[0]) == 5, "one register index per array element"
    (load,) = nodes(flat, tirx.BufferLoad)
    assert int(load.indices[0]) == 0
