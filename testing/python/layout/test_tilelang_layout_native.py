"""Semantic checks for layout algebra and the native annotation boundary."""

from itertools import product

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang import _ffi_api
from tilelang.layout import Fragment, Layout, PartialFragment
from tilelang.layout.payload import decode_layout_payload, encode_layout_payload
from tilelang.layout.swizzle import make_full_bank_swizzled_layout
from tilelang.layout.partial_fragment import make_fully_replicated_partial_fragment


def _ints(values):
    analyzer = sym.Analyzer()
    return tuple(int(analyzer.simplify(value)) for value in values)


def _round_trip(layout, kind):
    payload = encode_layout_payload(layout)
    result = decode_layout_payload(payload, kind)
    assert layout.is_equal(result)
    assert tvm_ffi.structural_equal(payload, encode_layout_payload(result))
    assert tvm_ffi.structural_hash(payload) == tvm_ffi.structural_hash(encode_layout_payload(result))
    return result


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8)])
def test_layout_payload_and_addresses(shape):
    def forward(*indices):
        if len(indices) == 2:
            return [indices[1] // 4, indices[0], indices[1] % 4]
        return [indices[0], indices[2] // 4, indices[1], indices[2] % 4]

    original = Layout(shape, forward)
    restored = _round_trip(original, "layout")
    for coordinate in product(*(range(n) for n in shape)):
        assert _ints(restored(*coordinate)) == tuple(forward(*coordinate))


def test_fragment_replication_keeps_owner_and_slot():
    # Each of two warps has an equal-value copy. Replication changes the owner,
    # while the per-thread storage index remains the logical row.
    fragment = Fragment(
        (4, 8),
        replicate=2,
        forward_fn=lambda i, j, rep: (rep * 32 + i * 8 + j, i),
    )
    restored = _round_trip(fragment, "fragment")
    for i, j, rep in product(range(4), range(8), range(2)):
        assert _ints(restored.map_forward_thread([rep, i, j])) == (rep * 32 + i * 8 + j,)
        assert _ints(restored.map_forward_index([rep, i, j])) == (i,)


def test_replicate_callback_var_projection_is_native():
    by_operand = Fragment((4,), replicate=2, forward_fn=lambda i, r: (4 * r + i, i))
    by_var = Fragment((4,), replicate=2, forward_fn=lambda i, r: (4 * r.var + i, i))
    assert by_operand.is_equal(by_var)


def test_partial_payload_preserves_addend_boundary_and_thread_range():
    partial = PartialFragment(
        (4,),
        replicate=8,
        combine=2,
        forward_fn=lambda i, rep: (i * 8 + rep, i),
    )
    payload = encode_layout_payload(partial)
    # The two constants are a begin and an extent, never an end coordinate.
    fields = list(payload.fields)
    fields[4] = ir.Tuple([ir.const(3), ir.const(64)])
    restored = decode_layout_payload(ir.Tuple(fields), "partial")
    assert isinstance(restored, PartialFragment)
    assert int(restored.combine_size) == 2
    assert int(restored.replicate_size) == 8
    assert int(restored.thread_range.begin) == 3
    assert int(restored.thread_range.extent) == 64
    assert not restored.is_equal(restored.as_post_collective())
    assert not restored.as_post_collective().is_equal(restored)
    other_combine = PartialFragment((4,), replicate=8, combine=4, forward_fn=lambda i, rep: (i * 8 + rep, i))
    assert not restored.is_equal(other_combine)
    _round_trip(restored, "partial")
    for i, rep in product(range(4), range(8)):
        owner = _ints(restored.map_forward_thread([rep, i]))[0]
        assert owner == i * 8 + rep
        assert (rep % int(restored.combine_size), rep // int(restored.combine_size)) == (rep % 2, rep // 2)


def test_payload_dynamic_shape_capture_and_specialization():
    n = ir.Var("n", "int32")
    layout = Layout((n, 8), lambda i, j: i * 8 + j)
    payload = encode_layout_payload(layout)
    free_vars = list(tirx.analysis.undefined_vars(payload))
    assert len(free_vars) == 1 and free_vars[0].same_as(n)
    # Both runs start from the same symbolic payload. The first specialization
    # must not mutate it or leak a coordinate from its lambda into the second.
    for extent in (3, 5):
        specialized = tvm_ffi.structural_map(
            payload, (ir.Var, lambda var, extent=extent: ir.const(extent) if var.same_as(n) else var), order="pre"
        )
        assert not list(tirx.analysis.undefined_vars(specialized))
        result = decode_layout_payload(specialized, "layout")
        assert _ints(result.get_input_shape()) == (extent, 8)
        for i, j in product(range(extent), range(8)):
            assert _ints(result(i, j)) == (i * 8 + j,)
    assert list(tirx.analysis.undefined_vars(payload))[0].same_as(n)


@pytest.mark.parametrize("kind", ["layout", "fragment", "partial"])
def test_payload_rejects_unknown_version(kind):
    payload = encode_layout_payload(Layout((4,), lambda i: i))
    with pytest.raises(ValueError, match="version"):
        decode_layout_payload(payload, kind, schema_version=2)


def test_payload_rejects_kind_and_addend_loss():
    payload = encode_layout_payload(PartialFragment((4,), replicate=4, combine=2, forward_fn=lambda i, r: (i * 4 + r, i)))
    with pytest.raises(ValueError, match="partial fragment"):
        decode_layout_payload(payload, "fragment")
    with pytest.raises(ValueError, match="plain layout"):
        decode_layout_payload(payload, "layout")
    with pytest.raises(ValueError, match="kind"):
        decode_layout_payload(payload, "unknown")


def test_native_bank_swizzle_matches_address_oracle():
    tensor = ir.Var("shared", tirx.TensorType((8, 64), "float16", scope="shared"))
    layout = make_full_bank_swizzled_layout(tensor)
    restored = _round_trip(layout, "layout")
    # An 8-row, 128-byte row bank swizzle uses the low 3 row bits to xor
    # the 8-element bank group, leaving the low 3 column bits unchanged.
    for row, column in product(range(8), range(64)):
        expected = row * 64 + ((column // 8) ^ row) * 8 + column % 8
        indices = _ints(restored(row, column))
        # The layout may split physical axes; compare the final address rather
        # than requiring a particular coordinate representation.
        address = 0
        for index, extent in zip(indices, _ints(restored.get_output_shape())):
            address = address * extent + index
        assert address == expected


def test_widening_noncontiguous_layout_rejected():
    layout = Layout((8,), lambda i: 2 * i)
    with pytest.raises(Exception, match="(?i)(contiguous|reinterpre|reshape|widen)"):
        layout.reshape((2,), rescale_num=8, rescale_den=32)


def test_layout_iter_var_ffi_uses_native_var():
    var = ir.Var("i", "int32")
    domain = ir.Range(0, 8)
    descriptor = _ffi_api.LayoutIterVar(domain, var)
    assert isinstance(descriptor.var, ir.Var)
    assert descriptor.var.same_as(var)


def test_reshape_preserves_physical_coordinates_and_source():
    source = Layout((3, 8), lambda i, j: (j // 4, i, j % 4))
    reshaped = source.reshape((6, 4))
    for i, j in product(range(6), range(4)):
        flat = i * 4 + j
        assert _ints(reshaped(i, j)) == (flat % 8 // 4, flat // 8, flat % 4)
    assert _ints(source.get_input_shape()) == (3, 8)


def test_reinterpret_preserves_storage_bits():
    source = Layout((4,), lambda i: i)
    bytes_layout = source.reshape((16,), rescale_num=32, rescale_den=8)
    for i in range(16):
        assert _ints(bytes_layout(i)) == (i,)
        assert _ints(source(i // 4))[0] * 32 + (i % 4) * 8 == i * 8
    restored = bytes_layout.reshape((4,), rescale_num=8, rescale_den=32)
    assert source.is_equal(restored)


def test_partial_collective_steps_exclude_copy_groups():
    # These are the node-only cases from the existing reducer-v2 suite. The
    # collective width and thread stride must retain their separate meanings.
    def steps(partial):
        return [tuple(map(int, step)) for step in _ffi_api.PartialFragment_combine_steps(partial)]

    wide = make_fully_replicated_partial_fragment((64,), 128)
    low = PartialFragment((32,), forward_thread_fn=lambda i, rep: (i % 16) * 8 + rep, replicate=8)
    high = PartialFragment((32,), forward_thread_fn=lambda i, rep: rep * 16 + i % 16, replicate=8)
    copied = PartialFragment((1,), forward_thread_fn=lambda i, rep: rep, replicate=128, combine=8)
    assert steps(wide) == [(128, 1)]
    assert steps(low) == [(8, 1)]
    assert steps(high) == [(128, 16)]
    assert steps(copied) == [(8, 1)]
    assert "(combine 8 x copy 16)" in repr(copied)
    for replica in range(128):
        assert _ints(copied.map_forward_thread([replica, 0])) == (replica,)
        assert replica % 8 + 8 * (replica // 8) == replica
    with pytest.raises(ValueError, match="divide"):
        PartialFragment((32,), forward_thread_fn=lambda i, rep: rep, replicate=8, combine=3)


def test_swizzle_accepts_native_load_and_region():
    tensor = ir.Var("shared", tirx.TensorType((8, 64), "float16", scope="shared"))
    expected = make_full_bank_swizzled_layout(tensor)
    load = ir.TensorLoad(tensor, [0, 0])
    region = ir.TensorRegion(tensor, [ir.Range(0, 8), ir.Range(0, 64)], tensor.ty)
    assert expected.is_equal(make_full_bank_swizzled_layout(load))
    assert expected.is_equal(make_full_bank_swizzled_layout(region))
