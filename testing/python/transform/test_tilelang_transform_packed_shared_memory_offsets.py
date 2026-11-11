# ruff: noqa
"""Packed sub-byte shared memory uses exactly one byte<->element conversion.

``MergeSharedMemoryAllocations`` stores the arena position of every merged
buffer as a byte offset.  Whenever that offset is turned back into an index of
the original typed buffer it has to go through ``ByteOffsetToElemOffset``: for
``float4_e2m1fn``/``int4``/``uint4`` one byte holds two logical elements, so
``DataType::bytes() * lanes()`` (one byte per element) would halve the offset
and alias two live buffers.

Both the aliasing (``handle_add_byte_offset``) and the non-aliasing
(index-shift) lowering of the pass are exercised here.  They run on hand-built
TIR, so no GPU or CUDA toolkit is needed.
"""

import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm import tirx
from tvm.script import tirx as T

# Arena alignment the pass applies to every merged allocation.
ALIGN = 16

# (dtype, elements, packed footprint in bytes), mirroring tile-ai/tilelang#2854.
CASES = [
    ("float4_e2m1fn", 16, 8),
    ("float4_e2m1fn", 127, 64),
    ("float4_e2m1fnx2", 16, 16),
    ("float4_e2m1fnx8", 16, 64),
    ("int4", 16, 8),
    ("int4", 127, 64),
    ("uint4", 127, 64),
    ("float16x8", 16, 256),
    ("int32x4", 16, 256),
    ("float16", 127, 254),
    ("float32", 127, 508),
]

WEBGPU = tvm.target.Target({"kind": "webgpu"})


def _align_up(value, align=ALIGN):
    return (value + align - 1) // align * align


def _elem_bits(dtype_name):
    dtype = tvm.DataType(dtype_name)
    return dtype.bits * dtype.lanes


def _two_live_buffers(dtype_name, elements, target=None):
    """Two simultaneously live shared buffers plus a pointer access to the second."""
    dtype = tvm.DataType(dtype_name)
    a = tirx.decl_buffer((elements,), dtype, name="A_shared", scope="shared.dyn")
    b = tirx.decl_buffer((elements,), dtype, name="B_shared", scope="shared.dyn")
    # A read of A after the write of B keeps both inside the same live range, so
    # the second buffer cannot reuse the first one's arena slot.
    access = tirx.tvm_access_ptr(tirx.type_annotation(dtype), b.data, 0, elements, 2)
    tx = tirx.Var("tx", "int32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.BufferStore(a, tirx.const(0, dtype), [0]),
            tirx.BufferStore(b, tirx.const(1, dtype), [0]),
            tirx.Evaluate(tirx.call_extern("void", "use", access, a[0])),
        ]
    )
    func = tirx.PrimFunc([], tirx.AttrStmt(tx, "thread_extent", 32, body))
    if target is not None:
        func = func.with_attr("target", target)
    return func


def _merge(func, disable_reuse=False):
    mod = tvm.IRModule({"main": func})
    return tl.transform.MergeSharedMemoryAllocations(disable_reuse=disable_reuse)(mod)


def _arena_bytes(mod):
    allocs = collect_nodes(mod, tirx.AllocBuffer)
    assert len(allocs) == 1, allocs
    buffer = allocs[0].buffer
    assert str(buffer.dtype) == "uint8"
    return int(buffer.shape[0])


def _alias_byte_offsets(mod):
    """Byte offsets of the ``handle_add_byte_offset`` alias bindings."""
    offsets = {}
    for bind in collect_nodes(mod, tirx.Bind):
        if isinstance(bind.value, tirx.Call):
            offsets[bind.var.name] = int(bind.value.args[1])
    return offsets


def _store_index(mod, name):
    for store in collect_nodes(mod, tirx.BufferStore):
        if store.buffer.name == name:
            assert len(store.indices) == 1, store
            return int(store.indices[0])
    raise AssertionError(f"no store to {name}")


def _access_ptr_offset(mod):
    calls = collect_calls(mod, op="tirx.tvm_access_ptr")
    assert len(calls) == 1, calls
    return int(calls[0].args[2])


@pytest.mark.parametrize("dtype_name,elements,packed_bytes", CASES)
def test_alias_byte_offsets_use_packed_footprint(dtype_name, elements, packed_bytes):
    """The arena reserves the packed footprint, not one byte per element."""
    mod = _merge(_two_live_buffers(dtype_name, elements), disable_reuse=True)

    expected_offset = _align_up(packed_bytes)
    offsets = _alias_byte_offsets(mod)
    assert offsets == {"A_shared": 0, "B_shared": expected_offset}
    assert _arena_bytes(mod) == 2 * expected_offset


@pytest.mark.parametrize("dtype_name,elements,packed_bytes", CASES)
def test_non_alias_index_shift_uses_packed_elements(dtype_name, elements, packed_bytes):
    """Without alias bindings the pass shifts indices in *packed* elements.

    The second buffer starts ``_align_up(packed_bytes)`` bytes into the arena,
    which is ``bytes * 8 / elem_bits`` logical elements of its own dtype.
    """
    mod = _merge(_two_live_buffers(dtype_name, elements, target=WEBGPU))

    expected_offset = _align_up(packed_bytes) * 8 // _elem_bits(dtype_name)
    assert expected_offset > 0
    assert _store_index(mod, "A_shared") == 0
    assert _store_index(mod, "B_shared") == expected_offset
    assert _access_ptr_offset(mod) == expected_offset


@pytest.mark.parametrize("dtype_name", ["float4_e2m1fn", "int4", "uint4"])
def test_sub_byte_buffers_do_not_alias(dtype_name):
    """A 4-bit buffer must not be shifted by one byte per element."""
    mod = _merge(_two_live_buffers(dtype_name, 16, target=WEBGPU))

    b_index = _store_index(mod, "B_shared")
    # Byte 16 of the arena is element 32 of a 4-bit buffer.  The buggy
    # one-byte-per-element conversion returned 16, i.e. byte 8, which overlaps
    # A_shared.
    assert b_index == 32
    assert b_index != 16


@pytest.mark.parametrize("dtype_name", ["float6_e3m2fn", "float6_e2m3fn"])
def test_dtype_that_neither_packs_nor_fills_a_byte(dtype_name):
    """Dtypes sized as one element per byte convert far offsets the same way."""
    mod = _merge(_two_live_buffers(dtype_name, 16, target=WEBGPU))

    assert _store_index(mod, "B_shared") == ALIGN


@pytest.mark.parametrize("dtype_name", ["float6_e3m2fnx4", "float6_e2m3fnx4"])
def test_vector_six_bit_dtype_reserves_whole_bytes(dtype_name):
    """Vectors that are sub-byte but not packed keep one byte per lane."""
    # 16 elements x 4 lanes x 1 byte = 64 bytes, so the second buffer starts at
    # byte 64 and element 16 -- not element 21, which is what dividing by the
    # tight 24-bit element size would give.
    mod = _merge(_two_live_buffers(dtype_name, 16), disable_reuse=True)
    assert _alias_byte_offsets(mod)["B_shared"] == 64

    mod = _merge(_two_live_buffers(dtype_name, 16, target=WEBGPU))
    assert _store_index(mod, "B_shared") == 16
    assert _access_ptr_offset(mod) == 16


def _mixed_width_alias_func():
    """Two live shared buffers, the second read through two pointer widths."""

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16,), "int4", scope="shared.dyn")
        B_shared = T.alloc_buffer((16,), "uint8", scope="shared.dyn")
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        # A_shared is laid out first, so B_shared starts at byte 16.
        T.evaluate(T.call_extern("void", "consume", T.address_of(A_shared[1])))
        T.evaluate(
            T.call_extern(
                "void",
                "writer",
                T.tvm_access_ptr(T.type_annotation("int4"), B_shared.data, 0, 1, 2),
            )
        )
        T.evaluate(
            T.call_extern(
                "void",
                "reader",
                T.tvm_access_ptr(T.type_annotation("uint8"), B_shared.data, 0, 1, 1),
                tx,
            )
        )

    return func


_CALL_EXTERN = tirx.op.Op.get("tirx.call_extern")
_TVM_STORAGE_SYNC = tirx.op.Op.get("tirx.tvm_storage_sync")


def _label(stmt):
    if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
        call = stmt.value
        if call.op.same_as(_TVM_STORAGE_SYNC):
            return "sync"
        if call.op.same_as(_CALL_EXTERN):
            return str(call.args[0].value)
    return type(stmt).__name__


def _labels_around_writer(mod):
    for seq in collect_nodes(mod, tirx.SeqStmt):
        labels = [_label(child) for child in seq.seq]
        if "writer" in labels and "reader" in labels:
            return labels
    raise AssertionError("expected the writer and reader in one SeqStmt")


def test_mixed_width_alias_keeps_barrier():
    """An `int4` and a `uint8` view of the same alias must not look disjoint.

    Both pointer accesses start at byte 16 of the arena, so another thread can
    read the half-byte this thread wrote and a barrier is required. The two
    views report element offsets 32 and 16; comparing those numbers directly
    proves a disjointness that does not exist.
    """
    mod = tvm.IRModule({"main": _mixed_width_alias_func()})
    mod = tl.transform.MergeSharedMemoryAllocations(disable_reuse=True)(mod)
    assert _alias_byte_offsets(mod) == {"A_shared": 0, "B_shared": ALIGN}

    mod = tl.transform.ThreadSync(["shared", "shared.dyn"])(mod)
    assert _labels_around_writer(mod) == ["consume", "writer", "sync", "reader"]


if __name__ == "__main__":
    tilelang.testing.main()
