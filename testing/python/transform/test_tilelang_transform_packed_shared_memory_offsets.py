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


if __name__ == "__main__":
    tilelang.testing.main()
