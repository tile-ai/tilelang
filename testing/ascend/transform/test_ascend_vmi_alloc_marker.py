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


def _slot(name="slot", dtype="int32", lanes=1, scope="local.var"):
    element = dtype if lanes == 1 else f"{dtype}x{lanes}"
    return tirx.decl_buffer((1,), element, name=name, scope=scope)


def _marker(entries):
    return {_MARKER: dict(entries)}


def _marker_blocks(ir):
    return [block for block in nodes(ir, tirx.SBlock) if _MARKER in block.annotations]


def _mapping(block):
    return block.annotations[_MARKER]


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
    for vf in vfs:
        (allocation,) = vf.alloc_buffers
        mapping = _mapping(vf)
        assert len(mapping) == 1
        (key,) = mapping.keys()
        assert key.same_as(allocation.data)
        assert str(mapping[key]) == _DATA
        stores, loads = _slot_uses(vf)
        assert stores and all(store.buffer.data.same_as(allocation.data) for store in stores)
        assert loads and all(load.buffer.data.same_as(allocation.data) for load in loads)
        data_vars.append(allocation.data)
    assert not data_vars[0].same_as(data_vars[1])

    # No surviving map names the original data Var: each clone maps only its
    # own fresh allocation.


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
