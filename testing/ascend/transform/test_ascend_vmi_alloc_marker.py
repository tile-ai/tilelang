"""Allocation markers for explicit mutable VMI data slots.

The frontend records ``tl.vmi.alloc_kind = {slot.data: "data"}`` on the
enclosing block.  ``NormalizePTOAllocMarkers`` rebuilds each map on the block
that actually owns the allocation.  Early explicit unroll must rebuild the
entry for every cloned allocation, and ``LowerOpaqueBlock`` must transfer the
kind onto the actual allocation before code generation consumes it.

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


def _slot(name="slot", dtype="int32", lanes=1, scope="local.var"):
    element = dtype if lanes == 1 else f"{dtype}x{lanes}"
    return tirx.decl_buffer((1,), element, name=name, scope=scope)


def _array(name="regs", shape=(2,), dtype="float32x64", scope="local"):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _marker(entries):
    return {_MARKER: dict(entries)}


def _marker_blocks(ir):
    return [block for block in nodes(ir, tirx.SBlock) if _MARKER in block.annotations]


def _mapping(block):
    return block.annotations[_MARKER]


def _seq_statements(stmt):
    if isinstance(stmt, tirx.SeqStmt):
        statements = []
        for child in stmt.seq:
            statements.extend(_seq_statements(child))
        return statements
    return [stmt]


def _module(body):
    return tvm.IRModule({"main": tirx.PrimFunc([], body)})


def _unroll(body):
    # Marker-carrying functions bind the PTO target and move the markers onto
    # the owning allocations before the early unroll clones them.
    mod = tirx.transform.BindTarget(_PTO_TARGET)(_module(body))
    mod = ascend.NormalizePTOAllocMarkers()(mod)
    return ascend.UnrollLoopSkipVF()(mod)["main"].body


def _lower_opaque(body):
    return tilelang.transform.LowerOpaqueBlock()(_module(body))["main"].body


def _loop(body_fn, *, extent=2, kind=tirx.ForKind.UNROLLED, explicit=True):
    var = tirx.Var("i", "int32")
    annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    return tirx.For(var, 0, extent, kind, body_fn(var), annotations=annotations)


def _vf(body, alloc_buffers=()):
    return tirx.SBlock([], [], [], "SIMD_VF", body, alloc_buffers=list(alloc_buffers))


def _slot_uses(ir):
    stores = [node for node in nodes(ir, tirx.BufferStore) if node.buffer.scope() == "local.var"]
    loads = [node for node in nodes(ir, tirx.BufferLoad) if node.buffer.scope() == "local.var"]
    return stores, loads


# ---------------------------------------------------------------------------
# Frontend registration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Ownership normalization moves the marker to the allocation owner
# ---------------------------------------------------------------------------
_PTO_TARGET = Target({"kind": "ascend", "keys": ["pto", "ascend"]})


def _normalize_module(body, target=_PTO_TARGET):
    mod = _module(body)
    if target is not None:
        mod = tirx.transform.BindTarget(target)(mod)
    return ascend.NormalizePTOAllocMarkers()(mod)


def _normalize(body, target=_PTO_TARGET):
    return _normalize_module(body, target)["main"].body


def _owner_of(ir, data):
    for block in nodes(ir, tirx.SBlock):
        if any(buffer.data.same_as(data) for buffer in block.alloc_buffers):
            return block
    raise AssertionError(f"no block owns the data Var {data}")


def test_normalize_moves_marker_to_allocation_owner():
    slot_a = _slot(name="duplicate")
    slot_b = _slot(name="duplicate")
    slot_c = _slot(name="plain_owner")
    slot_d = _slot(name="second")

    vf_one = _vf(
        tirx.BufferStore(slot_a, tirx.IntImm("int32", 1), [0]),
        alloc_buffers=[slot_a],
    )
    vf_two = _vf(
        tirx.SeqStmt(
            [
                tirx.BufferStore(slot_b, tirx.IntImm("int32", 2), [0]),
                tirx.BufferStore(slot_d, tirx.IntImm("int32", 4), [0]),
            ]
        ),
        alloc_buffers=[slot_b, slot_d],
    )
    plain = tirx.SBlock(
        [],
        [],
        [],
        "plain",
        tirx.BufferStore(slot_c, tirx.IntImm("int32", 3), [0]),
        alloc_buffers=[slot_c],
        annotations={"tl.owner_probe": "owner"},
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.SeqStmt([vf_one, vf_two, tirx.SBlockRealize([], True, plain)]),
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

    # Each owner keeps only its own marked allocation, with the original
    # data Var and the allocation still in place.
    owner_a = _owner_of(after, slot_a.data)
    assert owner_a.name_hint == "SIMD_VF"
    (allocation_a,) = owner_a.alloc_buffers
    assert allocation_a.data.same_as(slot_a.data)
    (key_a,) = _mapping(owner_a).keys()
    assert key_a.same_as(slot_a.data)
    assert str(_mapping(owner_a)[key_a]) == _DATA

    # The unmarked same-name twin in the other VF gains no entry by name;
    # the multi-slot owner maps exactly its own marked slot.
    owner_b = _owner_of(after, slot_b.data)
    owner_d = _owner_of(after, slot_d.data)
    assert owner_b.same_as(owner_d)
    (key_d,) = _mapping(owner_b).keys()
    assert key_d.same_as(slot_d.data)
    assert not key_d.same_as(slot_b.data)
    (allocation_b,) = [b for b in owner_b.alloc_buffers if b.data.same_as(slot_b.data)]
    assert allocation_b.data.same_as(slot_b.data)

    # A plain non-VF block, here inside an SBlockRealize, owns its own map.
    owner_c = _owner_of(after, slot_c.data)
    assert owner_c.name_hint == "plain"
    (allocation_c,) = owner_c.alloc_buffers
    assert allocation_c.data.same_as(slot_c.data)
    (key_c,) = _mapping(owner_c).keys()
    assert key_c.same_as(slot_c.data)
    (realize,) = nodes(after, tirx.SBlockRealize)
    assert realize.block.same_as(owner_c)
    # Unrelated annotations survive on the owner that receives the map.
    assert owner_c.annotations["tl.owner_probe"] == "owner"

    # Ownership metadata only: no allocation is materialized early and every
    # slot use still refers to an original data Var.
    assert not nodes(after, tirx.AllocBuffer)
    stores, _ = _slot_uses(after)
    assert len(stores) == 4
    originals = [slot_a.data, slot_b.data, slot_c.data, slot_d.data]
    for store in stores:
        assert any(store.buffer.data.same_as(data) for data in originals)


def test_normalize_array_marker_is_idempotent():
    regs = _array()
    vf = _vf(
        tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
        alloc_buffers=[regs],
    )
    root = tirx.SBlock([], [], [], "tilelang_root", vf, annotations=_marker({regs.data: _DATA_ARRAY}))
    once = _normalize_module(root)
    twice = ascend.NormalizePTOAllocMarkers()(once)

    # The rebuilt map already matches the recorded one, so a second run leaves
    # the function untouched instead of adding another marker authority.
    assert once["main"].body.same_as(twice["main"].body)

    (owner,) = _marker_blocks(once["main"].body)
    assert owner.name_hint == "SIMD_VF"
    (key,) = _mapping(owner).keys()
    assert key.same_as(regs.data)
    assert str(_mapping(owner)[key]) == _DATA_ARRAY


# ---------------------------------------------------------------------------
# Early explicit unroll clones the marker with the allocation Var
# ---------------------------------------------------------------------------
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
    data_vars = []
    out_indices = []
    for vf in vfs:
        (allocation,) = vf.alloc_buffers
        mapping = _mapping(vf)
        assert len(mapping) == 1
        (key,) = mapping.keys()
        assert key.same_as(allocation.data)
        assert str(mapping[key]) == _DATA
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

    # No surviving map names the original data Var: each clone maps only its
    # own fresh allocation.
    surviving_maps = [_mapping(block) for block in _marker_blocks(after)]
    assert len(surviving_maps) == len(vfs)
    for mapping in surviving_maps:
        assert not any(key.same_as(slot.data) for key in mapping)
    assert all(any(key.same_as(data) for data in data_vars) for mapping in surviving_maps for key in mapping)


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
    marked_vfs = [block for block in vfs if _MARKER in block.annotations]
    assert len(marked_vfs) == 2
    assert all(block.annotations[_MARKER] for block in marked_vfs)

    data_vars = []
    out_indices = []
    for block in marked_vfs:
        (allocation,) = block.alloc_buffers
        mapping = _mapping(block)
        (key,) = mapping.keys()
        assert key.same_as(allocation.data)
        assert str(mapping[key]) == _DATA_ARRAY
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

    # The unmarked twin allocations never receive an entry, and no surviving
    # map names either original data Var.
    twin_vars = []
    for block in vfs:
        if _MARKER in block.annotations:
            continue
        (twin,) = block.alloc_buffers
        twin_vars.append(twin.data)
        assert all(node.buffer.data.same_as(twin.data) for node in nodes(block, tirx.BufferStore) if node.buffer.scope() == "local")
        assert all(node.buffer.data.same_as(twin.data) for node in nodes(block, tirx.BufferLoad) if node.buffer.scope() == "local")
    assert len(twin_vars) == 2
    for block in _marker_blocks(after):
        for key in _mapping(block):
            assert not key.same_as(regs.data)
            assert not any(key.same_as(data) for data in twin_vars)


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


def test_unroll_rejects_map_outside_the_allocation_owner():
    # Skipping the ownership normalizer leaves the recording map on a block
    # that does not own the marked allocation.  The local clone rebuild
    # rejects that key with a diagnostic instead of silently dropping it.
    slot = _slot()

    def body(i):
        inner = _vf(
            tirx.BufferStore(slot, i, [0]),
            alloc_buffers=[slot],
        )
        return tirx.SBlock([], [], [], "carrier", inner, annotations=_marker({slot.data: _DATA}))

    root = tirx.SBlock([], [], [], "", _loop(body))
    with pytest.raises(tvm.error.InternalError, match="NormalizePTOAllocMarkers"):
        ascend.UnrollLoopSkipVF()(_module(root))


# ---------------------------------------------------------------------------
# LowerOpaqueBlock transfers the local marker map to the allocation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("realized", [False, True], ids=["direct", "realized"])
def test_lower_opaque_transfers_same_block_marker(realized):
    # Both entry points consume the block's own map: a directly lowered
    # SBlock and one reached through its SBlockRealize.
    slot = _slot()
    body = tirx.BufferStore(slot, tirx.IntImm("int32", 3), [0])
    block = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[slot],
        annotations=_marker({slot.data: _DATA}),
    )
    wrapped = tirx.SBlockRealize([], True, block) if realized else block
    after = _lower_opaque(wrapped)

    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert allocation.buffer.data.same_as(slot.data)
    assert str(allocation.annotations[_MARKER]) == _DATA
    assert not _marker_blocks(after)


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
def test_lower_opaque_rejects_malformed_annotation_container(make_marker, message):
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
        _lower_opaque(tirx.SBlockRealize([], True, block))


@pytest.mark.parametrize("realized", [False, True], ids=["direct", "realized"])
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
    block = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[regs, slot, plain],
        annotations=_marker({regs.data: _DATA_ARRAY, slot.data: _DATA}),
    )
    wrapped = tirx.SBlockRealize([], True, block) if realized else block
    after = _lower_opaque(wrapped)

    allocations = {node.buffer.name: node for node in nodes(after, tirx.AllocBuffer)}
    assert str(allocations["regs"].annotations[_MARKER]) == _DATA_ARRAY
    assert str(allocations["slot"].annotations[_MARKER]) == _DATA
    assert _MARKER not in allocations["plain"].annotations
    # The block map is consumed, not duplicated next to the allocations.
    assert not _marker_blocks(after)


@pytest.mark.parametrize("dtype", ["float64x64", "custom[tfloat32]x64"])
def test_lower_opaque_accepts_wide_and_tfloat_array_markers(dtype):
    # The shared allocation validator accepts every numeric width, including
    # float64 and tfloat32, and excludes only bool/handle.  This is the marker
    # contract, not instruction support for those element widths.
    regs = _array(dtype=dtype)
    block = tirx.SBlock(
        [],
        [],
        [],
        "",
        tirx.BufferStore(regs, tirx.BufferLoad(regs, [0]), [1]),
        alloc_buffers=[regs],
        annotations=_marker({regs.data: _DATA_ARRAY}),
    )
    after = _lower_opaque(tirx.SBlockRealize([], True, block))
    (allocation,) = [node for node in nodes(after, tirx.AllocBuffer)]
    assert str(allocation.annotations[_MARKER]) == _DATA_ARRAY
    assert str(allocation.buffer.dtype) == dtype


def test_lower_opaque_rejects_invalid_allocation_markers():
    def lowered(buffer, kind):
        block = tirx.SBlock(
            [],
            [],
            [],
            "",
            tirx.Evaluate(0),
            alloc_buffers=[buffer],
            annotations={_MARKER: {buffer.data: kind}},
        )
        return _lower_opaque(tirx.SBlockRealize([], True, block))

    regs = _array()
    with pytest.raises(tvm.error.InternalError, match="supports only"):
        lowered(regs, "mask_array")
    with pytest.raises(tvm.error.InternalError, match="may only mark a local.var allocation"):
        lowered(regs, _DATA)
    with pytest.raises(tvm.error.InternalError, match="may only mark a local allocation"):
        lowered(_slot(), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="numeric non-bool element type"):
        lowered(_array(name="bool_regs", dtype="boolx64"), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="multi-lane vector element dtype"):
        lowered(tirx.decl_buffer((2,), "float32", name="lane1", scope="local"), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="positive compile-time integer"):
        lowered(_array(name="dynamic", shape=(tirx.Var("n", "int32"),)), _DATA_ARRAY)
    with pytest.raises(tvm.error.InternalError, match="positive compile-time integer"):
        lowered(_array(name="empty", shape=(0,)), _DATA_ARRAY)


# ---------------------------------------------------------------------------
# Array markers survive flattening and keep one register index per element
# ---------------------------------------------------------------------------
def test_flatten_preserves_array_marker_and_index():
    regs = _array(shape=(2, 3))
    body = tirx.BufferStore(regs, tirx.BufferLoad(regs, [0, 0]), [1, 2])
    block = tirx.SBlock(
        [],
        [],
        [],
        "",
        body,
        alloc_buffers=[regs],
        annotations=_marker({regs.data: _DATA_ARRAY}),
    )
    lowered = _lower_opaque(tirx.SBlockRealize([], True, block))
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
