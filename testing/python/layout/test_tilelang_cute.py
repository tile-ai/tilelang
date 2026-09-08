import re

import pytest

import tilelang.testing
import tvm
from tvm import tirx

from tilelang.layout import Layout
from tilelang.layout import cute
from tilelang.layout.swizzle import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

tilelang.testing.set_random_seed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _swizzle(mode):
    """The (b_bits, m_base, s_shift) triple of a ComposedLayout's swizzle."""
    sw = mode.swizzle
    return (sw.b_bits, sw.m_base, sw.s_shift)


def _assert_struct(actual, shape, stride):
    """Assert a flat layout structurally equals make_layout(shape, stride)."""
    assert tvm.ir.structural_equal(actual, cute.make_layout(shape, stride=stride)), f"{actual} != {cute.make_layout(shape, stride=stride)}"


def _assert_same_fn(result, ref):
    """Assert two layouts are the same function: equal domain size and equal
    image at every coordinate. Works for hierarchical results that cannot be
    constructed with the flat make_layout."""
    assert cute.size(result) == cute.size(ref), f"size {cute.size(result)} != {cute.size(ref)}"
    for i in range(cute.size(ref)):
        assert result(i) == ref(i), f"at {i}: {result(i)} != {ref(i)}"


def _build_swizzled_layout(shape, intermediate_fn, b_bits, m_base, s_shift):
    """A single-output Layout whose forward index swizzles a bijective pow2
    intermediate address, applying the swizzle exactly as SwizzleNode::Apply:
    apply(x) = x ^ ((x & yyy) >> s_shift), with yyy = mask << (m_base + s_shift)
    and mask = (1 << b_bits) - 1."""

    def forward(*vars):
        addr = intermediate_fn(*vars)
        mask = (1 << b_bits) - 1
        yyy = mask << (m_base + s_shift)
        return addr ^ ((addr & yyy) >> s_shift)

    return Layout(shape, forward)


# ---------------------------------------------------------------------------
# to_python / from_python: round-trip nested shapes; a layout exposes its
# shape/stride as raw IntTuples.
# ---------------------------------------------------------------------------
def test_from_python_roundtrips_to_python():
    # from_python is the inverse of to_python for nested shapes.
    assert cute.to_python(cute.from_python([2, (3, 4)])) == (2, (3, 4))
    # A layout exposes its shape/stride as raw IntTuples for algebra.
    L = cute.make_layout((2, 3))
    assert L.shape == (2, 3)
    assert L.stride == (1, 2)


# ---------------------------------------------------------------------------
# IntTuple: construction, unpacking, and CuTe arithmetic (operator + / *).
# ---------------------------------------------------------------------------
def test_int_tuple_scalar_arithmetic():
    # Scalars add and multiply their values; 0 is the additive identity.
    assert cute.to_python(cute.from_python(3) + cute.from_python(4)) == 7
    assert cute.to_python(cute.from_python(3) * cute.from_python(4)) == 12
    assert cute.to_python(cute.from_python(5) + 0) == 5  # operand coercion + identity
    assert cute.to_python(2 * cute.from_python(6)) == 12  # __rmul__


def test_int_tuple_tuple_arithmetic():
    # Tuple + tuple adds component-wise; the shorter operand is zero-padded.
    a = cute.from_python([2, 3])
    assert cute.to_python(a + cute.from_python([4, 5])) == (6, 8)
    assert cute.to_python(a + cute.from_python([10])) == (12, 3)  # (2,3)+(10,) -> (12,3)


def test_int_tuple_scaled_basis_arithmetic():
    # ScaledBasis terms expand to ArithmeticTuples and add (CuTe
    # as_arithmetic_tuple): same axis sums into one slot, distinct axes spread.
    same = cute.from_python(cute.E(0)) + cute.from_python(2 * cute.E(0))
    assert cute.to_python(same) == (3,)  # 1@0 + 2@0 -> (1)+(2) -> (3)
    spread = cute.from_python(cute.E(0)) + cute.from_python(cute.E(1))
    assert cute.to_python(spread) == (1, 1)  # 1@0 + 1@1 -> (1)+(0,1) -> (1,1)


# ---------------------------------------------------------------------------
# product / flatten_to_tuple: leaf product of a raw shape; collapse a
# hierarchical shape/stride to a flat tuple of leaves.
# ---------------------------------------------------------------------------
def test_size_and_product():
    # size() is the domain of a Layout / ComposedLayout; product() is the leaf
    # product of a raw shape (flat, nested, or a scalar).
    assert cute.size(cute.make_layout((4, 8))) == 32
    assert cute.product((4, 8)) == 32
    assert cute.product((2, (2, 2))) == 8
    assert cute.product(5) == 5
    # Both thread PrimExprs for dynamic extents.
    n = tvm.tirx.Var("n", "int32")
    assert tvm.ir.structural_equal(cute.product((n, 4)), n * 4)
    assert tvm.ir.structural_equal(cute.size(cute.make_layout((n, 4), stride=(1, n))), n * 4)


def test_flatten_to_tuple_collapses_hierarchy():
    # A hierarchical composition result flattens to a flat tuple of leaves.
    R = cute.composition(cute.make_layout((8, 8), stride=(8, 1)), cute.make_layout((2, 4), stride=(1, 4)))
    assert cute.flatten_to_tuple(R.shape) == tuple(int(s) for s in cute.flatten(R).shape)
    assert len(cute.flatten_to_tuple(R.shape)) == len(cute.flatten_to_tuple(R.stride))


# ---------------------------------------------------------------------------
# Layout: rank / size / layout[idx] / flatten, layout(coord) evaluation,
# with_shape, and from_tilelang (recover a plain affine layout, else None).
# ---------------------------------------------------------------------------
def test_rank_size_getitem_flatten():
    L = cute.make_layout((4, 8), stride=(8, 1))
    assert cute.rank(L) == 2 and cute.size(L) == 32
    # layout[idx] is the i-th sublayout (a single mode unpacks to a scalar).
    assert L[0].shape == 4 and L[0].stride == 8
    assert L[1].shape == 8 and L[1].stride == 1
    # flatten of an already-flat layout is itself.
    assert tvm.ir.structural_equal(cute.flatten(L), L)


def test_eval_plain_is_scalar():
    # layout(coord): idx2crd (mode 0 fastest) dotted with the strides.
    L = cute.make_layout((4, 8), stride=(8, 1))
    assert [L(c) for c in range(4)] == [0, 8, 16, 24]
    assert L(4) == 1 and L(7) == 25 and isinstance(L(4), int)


def test_eval_tuple_coord():
    # crd2idx accepts a hierarchical coordinate (CuTe crd2idx_ttt), congruent to
    # the shape: layout((c0, c1)) == c0*stride0 + c1*stride1.
    L = cute.make_layout((4, 8), stride=(8, 1))
    assert L((1, 1)) == 9 and L((3, 7)) == 31
    assert all(L((c % 4, c // 4)) == L(c) for c in range(32))
    # Nested shape/stride with a nested coordinate.
    H = cute.make_layout((2, (2, 2)), stride=(1, (2, 4)))
    assert H((1, (1, 1))) == 1 + 2 + 4
    assert all(H(c) == H((c % 2, (c // 2 % 2, c // 4))) for c in range(8))


def test_eval_dynamic_shape_and_stride():
    # Neither shape nor stride need be constant: crd2idx threads PrimExprs.
    s = tvm.tirx.Var("s", "int32")
    L = cute.make_layout((4,), stride=(s,))
    assert tvm.ir.structural_equal(L(2), 2 * s)
    assert tvm.ir.structural_equal(L((3,)), 3 * s)
    # Dynamic extent: the mode is sized by a Var.
    n = tvm.tirx.Var("n", "int32")
    D = cute.make_layout((n, 4), stride=(1, n))
    assert tvm.ir.structural_equal(D((2, 1)), 2 + n)


def test_eval_basis_is_coordinate_tuple():
    # ScaledBasis strides make layout(coord) a coordinate: crd_k * (v@path)
    # lands v*crd_k in slot `path`, same-path contributions summing.
    ident = cute.make_identity_layout((4, 4))
    assert [ident(c) for c in range(16)] == [(c % 4, c // 4) for c in range(16)]
    # Permuted axes route mode k into the slot its basis names.
    perm = cute.make_layout((2, 3), stride=(cute.E(1), cute.E(0)))
    assert perm(5) == (2, 1)  # crd=(1,2): 1@slot1, 2@slot0
    # Same path on both modes sums into one slot; a single touched axis stays
    # a (1-)tuple, never a bare scalar.
    same = cute.make_layout((2, 2), stride=(cute.E(0), 2 * cute.E(0)))
    assert all(same(c) == (c,) for c in range(4))


def test_with_shape_splits_contiguous_run():
    # A flat layout reinterpreted over a 2D shape: (8):(1) with_shape (2,4) ->
    # the column-major linearization keeps function identity.
    L = cute.make_layout(8, stride=1)
    r = L.with_shape((2, 4))
    assert r.shape == (2, 4)
    for c0 in range(2):
        for c1 in range(4):
            assert r((c0, c1)) == L(c0 + 2 * c1)


def test_layout_from_tilelang_affine():
    L = cute.Layout.from_tilelang(Layout((16, 128), lambda i, j: i * 128 + j))
    assert L is not None
    _assert_struct(L, (16, 128), (128, 1))


def test_layout_from_tilelang_rejects_swizzle():
    swz = _build_swizzled_layout((64, 512), lambda i, j: i * 512 + j, 3, 3, 3)
    assert cute.Layout.from_tilelang(swz) is None


def test_parse_roundtrips_str():
    # IntTuple.parse / Layout.parse / Swizzle.parse invert str() exactly,
    # including nested tuples and innermost-first basis paths (E(1,0) is
    # spelled 1@0@1).
    for text in ("(((2,2,8),4),2):(((8@0,1@1,1@0),32@0),16@0)", "(128,64):(1@0,1@1)", "8:1"):
        assert str(cute.Layout.parse(text)) == text
    t = cute.from_python(cute.ScaledBasis(2, (1, 0)))
    assert str(cute.IntTuple.parse(str(t))) == str(t)
    assert str(cute.Swizzle.parse("Sw<3,4,3>")) == "Sw<3,4,3>"


# ---------------------------------------------------------------------------
# from_tilelang_hierarchical: recover a multi-output TileLang layout with each
# output axis rerouted onto its basis (the inverse of to_tilelang).
# ---------------------------------------------------------------------------
def test_from_tilelang_hierarchical_identity():
    r = cute.Layout.from_tilelang_hierarchical(Layout((128, 128), lambda i, j: [i, j]))
    assert r is not None
    assert tvm.ir.structural_equal(r, cute.make_identity_layout((128, 128)))


def test_from_tilelang_hierarchical_permuted_axes():
    # Rank-3 codomain: each input mode routes to the slot its output names.
    r = cute.Layout.from_tilelang_hierarchical(Layout((2, 3, 4), lambda i, j, k: [k, i, j]))
    assert r is not None
    assert tvm.ir.structural_equal(r, cute.make_layout((2, 3, 4), stride=(cute.E(1), cute.E(2), cute.E(0))))


def test_from_tilelang_hierarchical_single_output_is_flat_on_axis_zero():
    L = Layout((16, 128), lambda i, j: [i * 128 + j])
    flat = cute.Layout.from_tilelang(L)
    hier = cute.Layout.from_tilelang_hierarchical(L)
    assert hier is not None
    for c in range(0, 16 * 128, 97):
        assert hier(c) == (flat(c),)


def test_from_tilelang_hierarchical_conforms_axis_profiles():
    # Axis 0 sees j as one broadcast run while axis 1 splits j at 4; the
    # with_shape passes refine both to the common (4, 4) profile so the
    # strides can add leaf-wise.
    L = Layout((8, 16), lambda i, j: [i, (j // 4) * 8 + j % 4])
    r = cute.Layout.from_tilelang_hierarchical(L)
    assert r is not None
    assert tvm.ir.structural_equal(r, cute.make_layout((8, (4, 4)), stride=(cute.E(0), (cute.E(1), 8 * cute.E(1)))))
    for i in range(8):
        for j in range(16):
            assert r((i, j)) == (i, (j // 4) * 8 + j % 4)


def test_from_tilelang_hierarchical_roundtrips_to_tilelang():
    # A TMEM-shaped fragment (PTX Layout F datapath halves) survives the
    # to_tilelang -> from_tilelang_hierarchical round trip as a function.
    restride = cute.make_layout((128, 16384), stride=(cute.E(0), cute.E(1)))
    frag = cute.composition(restride, cute.make_layout(((16, 4), 32), stride=((1, 32), 128)))
    r = cute.Layout.from_tilelang_hierarchical(frag.to_tilelang())
    assert r is not None
    for i in range(0, 64, 7):
        for j in range(0, 32, 5):
            assert r((i, j)) == frag((i, j))


def test_from_tilelang_hierarchical_rejects_dependent_axes():
    # j drives both output axes: the leaf-wise stride sum would need a
    # coordinate-tuple stride, which no cute::Layout represents.
    assert cute.Layout.from_tilelang_hierarchical(Layout((4, 4), lambda i, j: [i + j, j])) is None


def test_from_tilelang_hierarchical_rejects_swizzled_axis():
    def swizzled(j):
        return j ^ ((j & (7 << 6)) >> 3)

    assert cute.Layout.from_tilelang_hierarchical(Layout((64, 512), lambda i, j: [swizzled(j), i])) is None


# ---------------------------------------------------------------------------
# coalesce: ported from CuTe pycute test_coalesce (flat cases). coalesce never
# changes layout(i), and a full collapse yields the scalar (1):(0).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,stride",
    [
        ((1,), (0,)),
        ((1,), (1,)),
        ((2, 4), (1, 2)),
        ((2, 4, 6), (1, 2, 8)),
        ((2, 4, 6), (1, 6, 2)),
        ((2, 1, 6), (1, 7, 2)),
        ((2, 1, 6), (4, 7, 8)),
        ((2, 4), (4, 1)),
        ((2, 4, 6), (24, 6, 1)),
        ((2, 1, 3), (2, 4, 4)),
        # Hierarchical operands (ported from CuTe pycute test_coalesce).
        ((2, (4, 6)), None),  # default (compact column-major) nested stride
        (((2, 2), (2, 2)), ((1, 4), (8, 32))),
    ],
)
def test_coalesce_preserves_function(shape, stride):
    L = cute.make_layout(shape, stride=stride)
    _assert_same_fn(cute.coalesce(L), L)


def test_coalesce_structural_results():
    # Contiguous bit-modes fuse to one; non-contiguous stay split; a fully
    # size-1 layout collapses to the scalar 1:0. A single surviving mode is
    # scalar-shaped (CuTe bw_coalesce returns Layout<NewShape,NewStride>).
    _assert_struct(cute.coalesce(cute.make_layout((2, 2, 2, 2, 2), stride=(1, 2, 4, 8, 16))), 32, 1)
    _assert_struct(cute.coalesce(cute.make_layout((8, 8), stride=(64, 1))), (8, 8), (64, 1))
    _assert_struct(cute.coalesce(cute.make_layout((1, 1), stride=(5, 7))), 1, 0)
    # A nested layout flattens, then fuses the contiguous (2,2):(1,4) prefix into
    # one mode while the gapped last mode stays split (CuTe pycute coalesce).
    _assert_struct(cute.coalesce(cute.make_layout(((2, 2), (2, 2)), stride=((1, 4), (8, 32)))), (2, 4, 2), (1, 4, 32))


# ---------------------------------------------------------------------------
# right_inverse: ported from CuTe pycute test_right_inverse (flat cases).
# Defining property: layout(inv(i)) == i for all i < size(inv).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,stride",
    [
        ((1,), (0,)),
        ((1, 1), (0, 0)),
        ((3, 7), (0, 0)),
        ((1,), (1,)),
        ((4,), (0,)),
        ((4,), (1,)),
        ((4,), (2,)),
        ((2, 4), (0, 2)),
        ((8, 4), (1, 8)),
        ((8, 4), (4, 1)),
        ((2, 4, 6), (1, 2, 8)),
        ((2, 4, 6), (4, 1, 8)),
        ((4, 2), (1, 16)),
        ((64, 8), (1, 64)),
    ],
)
def test_right_inverse(shape, stride):
    L = cute.make_layout(shape, stride=stride)
    inv = cute.right_inverse(L)
    for i in range(cute.size(inv)):
        assert L(inv(i)) == i


# ---------------------------------------------------------------------------
# composition: ported from CuTe pycute test_composition (flat cases).
# Defining property: (A o B)(i) == A(B(i)) for all i; size(A o B) == size(B).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ashape,astride,bshape,bstride",
    [
        ((1,), (0,), (1,), (0,)),
        ((1,), (1,), (1,), (1,)),
        ((4,), (1,), (4,), (1,)),
        ((4,), (2,), (4,), (1,)),
        ((4,), (0,), (4,), (1,)),
        ((4,), (1,), (2,), (2,)),
        ((4,), (2,), (2,), (2,)),
        ((12,), (1,), (4, 3), (3, 1)),
        ((12,), (1,), (2, 3), (2, 4)),
        ((12,), (2,), (4, 3), (1, 4)),
        ((4, 3), (1, 4), (12,), (1,)),
        ((4, 3), (1, 4), (6,), (2,)),
        ((4, 3), (3, 1), (12,), (1,)),
        ((4, 3), (3, 1), (6, 2), (2, 1)),
        ((4, 8), (8, 1), (8, 4), (1, 8)),
        ((2, 3, 4), (1, 2, 6), (6, 4), (1, 6)),
        ((64, 8), (8, 1), (8, 64), (1, 8)),
        ((8, 8), (8, 1), (2, 4), (1, 4)),
        ((4, 8, 2), (1, 4, 32), (8,), (2,)),
        # LHS contiguous only after coalescing (coalesce_x with a scalar
        # coprofile before the peel).
        ((2, 2), (1, 2), (4,), (1,)),
        ((4, 4), (1, 4), (8,), (1,)),
        ((3, 4), (1, 3), (6,), (1,)),
        # RHS reaches past the LHS domain via a trailing size-1 LHS mode (only
        # coalesce_x's shape-2 sentinel keeps the trailing slot).
        ((4, 1), (1, 0), (4,), (1,)),
        ((8, 1), (1, 0), (8,), (1,)),
        # Hierarchical operands (ported from CuTe pycute test_composition).
        ((8, 8), (8, 1), ((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))),
        (((4, 2),), ((1, 16),), (4, 2), (2, 1)),
        ((4, 8, 2), (2, 8, 1), (2, 2, 2), (1, 8, 2)),
        ((4, 6, 8), (1, 4, 7), (6,), (1,)),
        ((4, 6, 8, 10), (2, 3, 5, 7), (6,), (12,)),
        # Nested LHS composed with a flat RHS (the lhs tree is addressed by the
        # rhs's reach, not its own rank).
        (((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)), (8,), (4,)),
        # Default (compact column-major) LHS stride composed with a nested RHS.
        ((8, 8), None, ((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))),
        # Both operands rank-3, RHS reorders/strides into the LHS modes.
        ((4, 8, 2), (2, 8, 1), (4, 2, 2), (2, 8, 1)),
    ],
)
def test_composition_matches_bruteforce(ashape, astride, bshape, bstride):
    A = cute.make_layout(ashape, stride=astride)
    B = cute.make_layout(bshape, stride=bstride)
    R = cute.composition(A, B)
    # (A o B)(i) == A(B(i)) for all i; size(A o B) == size(B).
    assert cute.size(R) == cute.size(B)
    for i in range(cute.size(B)):
        assert R(i) == A(B(i))


def test_composition_structural_results():
    # Structural checks for results whose RHS reaches past the LHS domain (CuTe
    # extends the last mode linearly, so the brute-force oracle -- which wraps --
    # cannot validate these): (4):(1) o (4):(2) == (4):(2).
    _assert_struct(cute.composition(cute.make_layout((4,), stride=(1,)), cute.make_layout((4,), stride=(2,))), (4,), (2,))
    # A row-major LHS sub-block: (4,3):(3,1) o (12):(1) flattens to (4,3):(3,1).
    R = cute.composition(cute.make_layout((4, 3), stride=(3, 1)), cute.make_layout((12,), stride=(1,)))
    assert cute.flatten_to_tuple(R.shape) == (4, 3)
    assert cute.flatten_to_tuple(R.stride) == (3, 1)


def test_composition_scaled_basis_routing():
    # A ScaledBasis RHS routes each result mode into the LHS axis its path
    # names (basis_get), independent of mode order. E(1),E(0) swaps axes, so the
    # result strides follow the routing -> structurally [4,8]:[100,10].
    A = cute.make_layout((8, 4), stride=(10, 100))
    B = cute.make_layout((4, 8), stride=(cute.E(1), cute.E(0)))
    _assert_struct(cute.composition(A, B), (4, 8), (100, 10))
    # The identity RHS over [4,4] selects the [4,4] sub-block of A.
    A2 = cute.make_layout((16, 16), stride=(1, 16))
    R = cute.composition(A2, cute.make_identity_layout((4, 4)))
    _assert_struct(R, (4, 4), (1, 16))


def test_composition_scaled_basis_dynamic_strides():
    # Identity RHS routes each mode into the matching LHS axis, so the result
    # strides are the LHS's dynamic strides s_m, s_n (the is_scaled_basis branch).
    s_m, s_n = tvm.tirx.Var("s_m", "int32"), tvm.tirx.Var("s_n", "int32")
    R = cute.composition(cute.make_layout((16, 16), stride=(s_m, s_n)), cute.make_identity_layout((4, 4)))
    assert R.shape == (4, 4)
    sm, sn = R.stride
    assert tvm.ir.structural_equal(sm, s_m) and tvm.ir.structural_equal(sn, s_n)


def test_right_inverse_composition_smem_to_gmem():
    # The TMA building block: composing a GMEM layout with the inverse of a SMEM
    # layout yields a SMEM-address -> GMEM-address map.
    smem = cute.make_layout((8, 64), stride=(64, 1))
    gmem = cute.make_layout((8, 64), stride=(1024, 1))
    composite = cute.composition(gmem, cute.right_inverse(smem))
    for c in range(cute.product((8, 64))):
        assert composite(smem(c)) == gmem(c)


def test_tma_box_validity_via_composite():
    # Coalesce(GMEM o RightInverse(SMEM)) decides TMA-expressibility: its
    # innermost stride must be 1. Row-major SMEM passes; transposed fails.
    gmem = cute.make_layout((8, 64), stride=(256, 1))
    ok = cute.coalesce(cute.composition(gmem, cute.right_inverse(cute.make_layout((8, 64), stride=(64, 1)))))
    bad = cute.coalesce(cute.composition(gmem, cute.right_inverse(cute.make_layout((8, 64), stride=(1, 8)))))
    assert list(ok.stride)[0] == 1
    assert list(bad.stride)[0] != 1


# ---------------------------------------------------------------------------
# Layout algebra ported from CuTe for the GMMA descriptor analysis:
# filter / cosize / complement / logical_divide / logical_product. Reference
# values follow CuTe layout.hpp and its core logical_product unit test.
# ---------------------------------------------------------------------------
def test_filter_drops_trivial_modes():
    # filter = coalesce(filter_zeros): drop stride-0 and size-1 modes.
    _assert_struct(cute.filter(cute.make_layout((4, 1, 2), stride=(1, 0, 8))), (4, 2), (1, 8))
    _assert_struct(cute.filter(cute.make_layout((1, 1), stride=(5, 7))), 1, 0)


def test_filter_keeps_dynamic_modes():
    s = tvm.tirx.Var("s", "int32")
    # A const-0 stride mode is dropped; the dynamic-stride mode survives.
    _assert_struct(cute.filter(cute.make_layout((4, 1, 8), stride=(s, 0, 1))), (4, 8), (s, 1))


def test_cosize_matches_cute():
    # cosize = 1 + sum_i (shape_i - 1) * |stride_i|.
    assert cute.cosize(cute.make_layout((4, 2), stride=(1, 8))) == 12
    assert cute.cosize(cute.make_layout((8, 8), stride=(8, 1))) == 64
    assert cute.cosize(cute.make_layout((64, 8), stride=(8, 1))) == 512


def test_coshape_basis_strides_is_per_axis():
    # ScaledBasis strides make the codomain a coordinate space; coshape gives
    # the per-axis extents and cosize their product (CuTe cosize ==
    # size(coshape)).
    L = cute.make_layout((128, 32), stride=(cute.E(0), cute.E(1)))
    assert cute.coshape(L) == (128, 32)
    assert cute.cosize(L) == 128 * 32
    # PTX Layout F atom (interleaved half datapaths): only 112 datapaths are
    # touched (max image 96 + 15), and 32 columns.
    F = cute.make_layout(((16, 4), 32), stride=((cute.E(0), 32 * cute.E(0)), cute.E(1)))
    assert cute.coshape(F) == (112, 32)
    # A plain layout's coshape is its scalar codomain extent.
    assert cute.coshape(cute.make_layout((4, 2), stride=(1, 8))) == 12


def test_cosize_dynamic_is_symbolic():
    s = tvm.tirx.Var("s", "int32")
    # cosize = 1 + (4-1)*|s| + (8-1)*1 = 3*|s| + 8 ; stays a PrimExpr, and the
    # |stride| term emits CuTe's abs(stride). Check by substituting s.
    got = cute.cosize(cute.make_layout((4, 8), stride=(s, 1)))
    assert isinstance(got, tirx.PrimExpr)
    ana = tvm.arith.Analyzer()
    for sv, expect in [(5, 3 * 5 + 8), (-5, 3 * 5 + 8)]:
        sub = tvm.tirx.stmt_functor.substitute(got, {s: tvm.tirx.const(sv, "int32")})
        assert int(ana.simplify(sub)) == expect


def test_left_inverse_matches_cute():
    # Official left_inverse results (compiled against the CuTe headers):
    # a permuted layout inverts across modes, and a broadcast/batch-sliced
    # layout gets a leading stride-0 slot absorbing the skipped strides.
    _assert_struct(cute.left_inverse(cute.make_layout((4, 8))), 32, 1)
    _assert_struct(cute.left_inverse(cute.make_layout((4, 8), stride=(8, 1))), (8, 4), (4, 1))
    _assert_struct(cute.left_inverse(cute.make_layout(8, stride=2)), (2, 8), (0, 1))
    _assert_struct(cute.left_inverse(cute.make_layout((3, 8192), stride=(16384, 1))), (16384, 3), (3, 1))
    _assert_struct(
        cute.left_inverse(cute.make_layout((2, 3, 4), stride=(12, 1, 3))),
        (12, 2),
        (2, 1),
    )
    # result(layout(i)) == i on a permuted case.
    L = cute.make_layout((4, 3), stride=(1, 8))
    li = cute.left_inverse(L)
    for i in range(4):
        for j in range(3):
            assert li(L((i, j))) == i + 4 * j


def test_complement_matches_cute():
    # complement(layout, cotarget): the gaps the layout does not address.
    _assert_same_fn(cute.complement(cute.make_layout(2, stride=1), 16), cute.make_layout(8, stride=2))
    # 4:1 within 24 -> rest is 6:4.
    _assert_same_fn(cute.complement(cute.make_layout(4, stride=1), 24), cute.make_layout(6, stride=4))
    # Structural: a single surviving mode is SCALAR-shaped (CuTe bw_coalesce
    # returns Layout<NewShape,NewStride>, not a 1-tuple).
    _assert_struct(cute.complement(cute.make_layout(2, stride=1), 16), 8, 2)
    _assert_struct(cute.complement(cute.make_layout(4, stride=1), 24), 6, 4)


def test_complement_rank1_dynamic():
    s = tvm.tirx.Var("s", "int32")
    # complement((4):(s), 1024) = coalesce((s, ceil_div(1024, 4s)):(1, 4s)).
    comp = cute.complement(cute.make_layout(4, stride=s), 1024)
    assert comp.shape[0] == s and comp.stride[0] == 1
    # rest stride = 4*s ; rest extent = ceil_div(1024, 4*s) (symbolic).
    ana = tvm.arith.Analyzer()
    assert ana.simplify(comp.stride[1] - 4 * s) == 0


def test_complement_rank_gt1_dynamic_rejected():
    s = tvm.tirx.Var("s", "int32")
    # CuTe static_assert: dynamic-stride complement only for rank-1 layouts.
    with pytest.raises(Exception, match="rank-1"):
        cute.complement(cute.make_layout((4, 8), stride=(s, 1)), 1024)


def test_logical_divide_matches_cute():
    # logical_divide splits a mode into (tile, rest).
    _assert_same_fn(
        cute.logical_divide(cute.make_layout(8, stride=1), cute.make_layout(2, stride=1)), cute.make_layout((2, 4), stride=(1, 2))
    )
    _assert_same_fn(
        cute.logical_divide(cute.make_layout(8, stride=1), cute.make_layout(8, stride=1)), cute.make_layout((8, 1), stride=(1, 0))
    )
    # Structural: the result is FLAT (tile, rest), congruent to (_1, _1) -- the
    # complement's single mode is scalar, so no spurious nesting (mirrors CuTe).
    _assert_struct(cute.logical_divide(cute.make_layout(8, stride=1), cute.make_layout(2, stride=1)), (2, 4), (1, 2))
    _assert_struct(cute.logical_divide(cute.make_layout(8, stride=1), cute.make_layout(8, stride=1)), (8, 1), (1, 0))
    # Strided base (GMMA MN mode 128:4 divided by the W=8 tile) stays flat.
    _assert_struct(cute.logical_divide(cute.make_layout(128, stride=4), cute.make_layout(8, stride=1)), (8, 16), (4, 32))
    assert cute.congruent(cute.logical_divide(cute.make_layout(128, stride=4), cute.make_layout(8, stride=1)).shape, (1, 1))


def test_logical_divide_dynamic_layout_stride():
    s = tvm.tirx.Var("s", "int32")
    # Static shape (so size is concrete) but a dynamic stride: division still
    # works because complement is on the (static) tiler and composition handles
    # the dynamic layout stride. (8):(s) by (2):(1) -> (2, 4):(s, 2s).
    res = cute.logical_divide(cute.make_layout(8, stride=s), cute.make_layout(2, stride=1))
    ana = tvm.arith.Analyzer()
    # Result is hierarchical: shape (2, (4,)), stride (s, (2s,)).
    assert ana.simplify(res(0) - 0) == 0
    assert ana.simplify(res(1) - s) == 0  # next tile element: stride s
    assert ana.simplify(res(2) - 2 * s) == 0  # next rest element: stride 2s


def test_compatible_is_directional_and_supports_symbolic_extents():
    n = tvm.tirx.Var("n", "int32")

    # A terminal in the profile may cover an arbitrarily nested subtree.
    assert cute.compatible((2, 3), ((1, 2), 3))
    assert not cute.compatible(((1, 2), 3), (2, 3))
    assert not cute.compatible((2, 3), (6,))

    # Dynamic terminals are compared by provable value, not expression form.
    assert cute.compatible((n + n, 4), ((n, 2), (2, 2)))


@pytest.mark.parametrize(
    ("block", "tiler"),
    [
        (cute.make_layout(1, stride=0), cute.make_layout(1, stride=0)),
        (cute.make_layout(1, stride=1), cute.make_layout(1, stride=0)),
        (cute.make_layout(1, stride=0), cute.make_layout(1, stride=1)),
        (cute.make_layout(1, stride=1), cute.make_layout(1, stride=1)),
        (cute.make_layout(3, stride=1), cute.make_layout(4, stride=0)),
        (cute.make_layout(3, stride=0), cute.make_layout(4, stride=1)),
        (cute.make_layout(3, stride=0), cute.make_layout(4, stride=0)),
        (cute.make_layout(3, stride=2), cute.make_layout(4, stride=1)),
        (cute.make_layout((3, 1)), cute.make_layout((2, 4))),
        (cute.make_layout((2, 4)), cute.make_layout(3)),
        (cute.make_layout((8, (2, 2))), cute.make_layout(4, stride=2)),
        (cute.make_layout((2, 2)), cute.make_layout((3, 3), stride=(3, 1))),
        (cute.make_layout(3, stride=32), cute.make_layout(32)),
        (cute.make_layout(3, stride=2), cute.make_layout(4)),
        (cute.make_layout(3, stride=32), cute.make_layout(128)),
        (cute.make_layout(3, stride=32), cute.make_layout((8, 8))),
        (cute.make_layout(3, stride=32), cute.make_layout((8, 8), stride=(8, 1))),
        (cute.make_layout(((4, 2),), stride=((1, 16),)), cute.make_layout((4, 4))),
        (cute.make_layout(((4, 2),), stride=((1, 16),)), cute.make_layout((4, 2), stride=(2, 1))),
        (
            cute.make_layout(((2, 2), (2, 2)), stride=((1, 4), (8, 32))),
            cute.make_layout((2, 2), stride=(1, 2)),
        ),
        (
            cute.make_layout(((2, 2), (2, 2)), stride=((1, 4), (8, 32))),
            cute.make_layout((2, 2), stride=(2, 1)),
        ),
        (cute.make_layout(((4, 6),), stride=((1, 6),)), cute.make_layout(3)),
    ],
)
def test_logical_product_matches_cutlass_core_cases(block, tiler):
    # CUTLASS test/unit/cute/core/logical_product.cpp specifies these two core
    # properties: the result is rank-2, preserves the block as mode 0, and its
    # repeated mode is shape-compatible with the tiler.
    result = cute.logical_product(block, tiler)
    assert cute.rank(result) == 2
    assert tvm.ir.structural_equal(result[0], block)
    assert cute.compatible(tiler.shape, result[1].shape)


def test_logical_product_tmem_tiling():
    # The exact construction used by CUTLASS tmem_frg::make: repeat a 128x16
    # virtual TMEM atom over two M tiles and four K/N tiles, M-first.
    atom = cute.make_layout((128, 16))
    outer = cute.make_layout((2, 4))
    result = cute.logical_product(atom, outer)
    _assert_struct(result, ((128, 16), (2, 4)), ((1, 128), (2048, 4096)))

    tiled = cute.tiled_product(atom, outer)
    _assert_struct(tiled, ((128, 16), 2, 4), ((1, 128), 2048, 4096))
    for m_tile in range(2):
        for n_tile in range(4):
            assert tiled(((0, 0), m_tile, n_tile)) == result(((0, 0), (m_tile, n_tile)))


def test_logical_product_nested_strided_layout_exactly():
    block = cute.make_layout(((4, 2),), stride=((1, 16),))
    tiler = cute.make_layout((4, 2), stride=(2, 1))
    result = cute.logical_product(block, tiler)

    # Exact output for the corresponding CUTLASS core test case.  This covers
    # both a hierarchical block and a nontrivially strided tiler.
    _assert_struct(
        result,
        (((4, 2),), ((2, 2), 2)),
        (((1, 16),), ((8, 32), 4)),
    )


def test_tiled_product_unpacks_complement_split_modes():
    # A strided block can split one scalar tiler into two residual modes.
    # Official tiled_product slices with repeat<R1>(_), and repeat<1>(_) is
    # plain _, so only a rank >= 2 rest mode unpacks.  A scalar-leaf block's
    # complement modes land directly in the rest (rank 2 -> unpacked); a
    # tuple-wrapped block keeps them nested one level deeper (rank-1 rest ->
    # identical to the zipped product).  Both pinned against the compiled
    # official results.
    block = cute.make_layout(3, stride=32)
    tiler = cute.make_layout(128)
    logical = cute.logical_product(block, tiler)
    _assert_struct(logical, (3, (32, 4)), (32, (1, 96)))
    tiled = cute.tiled_product(block, tiler)
    _assert_struct(tiled, (3, 32, 4), (32, 1, 96))
    for x in range(3):
        for y0 in range(32):
            for y1 in range(4):
                assert tiled((x, y0, y1)) == logical((x, (y0, y1)))
    # A tuple-wrapped TILER keeps the composed rest congruent to its rank-1
    # profile, so the rest stays wrapped and tiled == zipped (official
    # repeat<1>(_) slicing); the scalar-leaf tiler above unpacked because
    # composition against a scalar profile leaves the split modes flat.
    wrapped_tiler = cute.Layout.parse("(128):(1)")
    tiled_wrapped = cute.tiled_product(block, wrapped_tiler)
    assert tvm.ir.structural_equal(tiled_wrapped, cute.logical_product(block, wrapped_tiler))
    assert str(tiled_wrapped) == "(3,((32,4))):(32,((1,96)))"


def test_blocked_product_zips_block_and_tiler_modes():
    # CuTe layout.hpp blocked_product: mode i of the result is
    # ((block_i, rest_i)), so a flat per-mode coordinate covers
    # block extent x tiler extent along that mode.
    block = cute.make_layout((2, 5), stride=(5, 1))
    tiler = cute.make_layout((3, 4))
    result = cute.blocked_product(block, tiler)
    _assert_struct(result, ((2, 3), (5, 4)), ((5, 10), (1, 30)))
    logical = cute.logical_product(block, tiler)
    for i in range(6):
        for j in range(20):
            assert result((i, j)) == logical(((i % 2, j % 5), (i // 2, j // 5)))


def test_blocked_product_pads_rank_mismatch():
    # A tiler of higher rank appends fresh modes (block padded with (1):(0)),
    # exactly CuTe's append<R>() padding.
    block = cute.make_layout((128, 16))
    tiler = cute.make_layout((1, 1, 2, 4))
    result = cute.blocked_product(block, tiler)
    assert cute.rank(result) == 4
    assert result((0, 0, 1, 0)) == 2048
    assert result((0, 0, 0, 1)) == 4096
    assert result((127, 15, 1, 3)) == 127 + 15 * 128 + 2048 + 3 * 4096
    # Exact structural match with the official result (compiled against the
    # CuTe headers): the tiler's size-1 modes carry stride 0, so the padded
    # block modes pair with stride-0 rest modes, not leftover strides.
    _assert_struct(
        result,
        ((128, 1), (16, 1), (1, 2), (1, 4)),
        ((1, 0), (128, 0), (0, 2048), (0, 4096)),
    )


def test_restrict_slices_basis_coordinate_layout():
    # A ScaledBasis-strided layout maps to (datapath, column) coordinates.
    # restrict returns the region origin's coordinates plus the sliced
    # sublayout, and the affine identity holds componentwise.
    restride = cute.make_layout((128, 16384), stride=(cute.E(0), cute.E(1)))
    frag = cute.composition(restride, cute.make_layout(((16, 4), 32), stride=((1, 32), 128)))
    assert frag((16, 0)) == (32, 0)
    offset, tile = cute.restrict(frag, [tvm.ir.Range.from_min_extent(0, 64), tvm.ir.Range.from_min_extent(8, 24)])
    assert offset == (0, 8)
    assert tile((0, 3)) == (0, 3)
    for i in range(0, 64, 7):
        for j in range(0, 24, 5):
            assert frag((i, 8 + j)) == tuple(base + step for base, step in zip(offset, tile((i, j))))


# ---------------------------------------------------------------------------
# make_layout / make_column_major / make_row_major / make_identity_layout, and
# the make_layout([layout, ...]) concat form.
# ---------------------------------------------------------------------------
def test_make_layout_default_stride_is_column_major():
    # Omitting stride yields the compact column-major stride (mode 0 fastest).
    assert tvm.ir.structural_equal(cute.make_layout((4, 8)), cute.make_column_major_layout((4, 8)))
    _assert_struct(cute.make_layout((4, 8)), (4, 8), (1, 4))


def test_make_identity_layout_strides_are_unit_basis():
    # Each identity stride is the unit basis E<k>.
    L = cute.make_layout((4, 4), stride=(cute.E(0), cute.E(1)))
    assert tvm.ir.structural_equal(L, cute.make_identity_layout((4, 4)))
    for k, s in enumerate(L.stride):
        assert isinstance(s, cute.ScaledBasis) and s.value == 1 and s.mode == (k,)


def test_make_layout_nested_column_row_identity():
    # Column/row-major and identity over a nested shape produce congruent nested
    # strides (CuTe compact_col_major / compact_row_major / make_identity_layout).
    assert tvm.ir.structural_equal(cute.make_column_major_layout((2, (2, 2))), cute.make_layout((2, (2, 2)), stride=(1, (2, 4))))
    assert tvm.ir.structural_equal(cute.make_row_major_layout((2, (2, 2))), cute.make_layout((2, (2, 2)), stride=(4, (2, 1))))
    # Identity maps each linear coord to its full nested idx2crd coordinate.
    I = cute.make_identity_layout((2, (2, 2)))
    assert I.shape == (2, (2, 2))
    assert [I(c) for c in range(8)] == [(c % 2, (c // 2 % 2, c // 4)) for c in range(8)]
    # Dynamic extents thread through the makers.
    n = tvm.tirx.Var("n", "int32")
    assert tvm.ir.structural_equal(cute.make_column_major_layout((n, 4)), cute.make_layout((n, 4), stride=(1, n)))


def test_scaled_basis_and_int_expr_strides():
    # A non-unit ScaledBasis stride round-trips its scale and mode.
    (sb,) = cute.make_layout((2,), stride=(64 * cute.E(1),)).stride
    assert isinstance(sb, cute.ScaledBasis) and sb.value == 64 and sb.mode == (1,)
    # A dynamic (PrimExpr) stride round-trips structurally.
    s = tvm.tirx.Var("s", "int32")
    (leaf,) = cute.make_layout((8,), stride=(s,)).stride
    assert tvm.ir.structural_equal(leaf, s)


def test_make_layout_concat():
    a = cute.make_layout(64, stride=1)
    b = cute.make_layout(4, stride=64)
    cat = cute.make_layout([a, b])
    assert cat.shape == (64, 4)
    assert cat.stride == (1, 64)


# ---------------------------------------------------------------------------
# ComposedLayout.recast: shift m_base by log2(old) - log2(new); b_bits /
# s_shift preserved (a linear layout recast is a no-op).
# ---------------------------------------------------------------------------
def test_recast_method():
    # Recast shifts m_base by log2(old) - log2(new); b_bits / s_shift preserved.
    mode = cute.ComposedLayout.from_tilelang(_build_swizzled_layout((64, 512), lambda i, j: (j % 64) + i * 64 + (j // 64) * 4096, 3, 3, 3))
    assert _swizzle(mode) == (3, 3, 3)
    assert _swizzle(mode.recast(16, 8)) == (3, 4, 3)  # smaller elem -> m_base += 1
    assert _swizzle(mode.recast(32, 8)) == (3, 5, 3)  # m_base += 2
    assert _swizzle(mode.recast(16, 16)) == (3, 3, 3)  # same width -> no-op


def test_recast_linear_is_noop():
    mode = cute.ComposedLayout.from_tilelang(Layout((16, 128), lambda i, j: i * 128 + j))
    assert not mode.swizzle.is_swizzled
    assert _swizzle(mode.recast(32, 8)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# ComposedLayout.from_tilelang: recover a Swizzle over an affine layout.
# Bank swizzles are Sw<b, 4, 3> in byte space (b = 1/2/3 for 32/64/128B).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "maker,b_bits",
    [
        (make_quarter_bank_swizzled_layout, 1),
        (make_half_bank_swizzled_layout, 2),
        (make_full_bank_swizzled_layout, 3),
    ],
)
def test_real_bank_swizzles(maker, b_bits):
    # continuous = vector_size * 2^b columns (float16 => vector_size 8).
    buf = tirx.decl_buffer((8, 8 * (1 << b_bits)), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(maker(buf), buf)
    assert mode is not None and _swizzle(mode) == (b_bits, 4, 3)


@pytest.mark.parametrize("dtype", ["int8", "float16", "float32"])
def test_bank_swizzle_mbase_is_byte_invariant(dtype):
    # m_base is 4 in byte space for every dtype (the swizzle acts at a fixed
    # 128-bit vector granularity): int8 (4+0), float16 (3+1), float32 (2+2).
    vector_size = 128 // int(tvm.DataType(dtype).bits)
    buf = tirx.decl_buffer((8, vector_size * 8), dtype, name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(make_full_bank_swizzled_layout(buf), buf)
    assert mode is not None and _swizzle(mode) == (3, 4, 3)


@pytest.mark.parametrize("lead", [2, 3, 5])
def test_expanded_layout_with_nonpow2_leading_dim(lead):
    # ExpandLayout2D prepends a (possibly non-pow2) batch dim; the trailing
    # swizzled tile is still detected.
    layout = make_full_bank_swizzled_layout(tirx.decl_buffer((8, 64), "float16", name="A", scope="shared")).expand([lead])
    buf = tirx.decl_buffer((lead, 8, 64), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None and _swizzle(mode) == (3, 4, 3)


# make_mma_swizzle_layout is the canonical bank swizzle Sw<b, 4, 3>: the shift
# stays 3 (8-row period) and b follows the 16B-vector count per row -- full(8)/
# half(4)/quarter(2) -> b=3/2/1. Wider rows keep b=3 (the block index tiles into
# an outer mode), so every case stays hardware-expressible.
@pytest.mark.parametrize(
    "dtype,cols,b_bits",
    [
        ("float16", 32, 2),
        ("float16", 64, 3),
        ("float16", 128, 3),
        ("float16", 256, 3),
        ("float32", 32, 3),
        ("float32", 64, 3),
        ("int8", 64, 2),
        ("int8", 128, 3),
    ],
)
def test_mma_swizzle_layout(dtype, cols, b_bits):
    buf = tirx.decl_buffer((16, cols), dtype, name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(make_mma_swizzle_layout(buf), buf)
    assert mode is not None and _swizzle(mode) == (b_bits, 4, 3)


@pytest.mark.parametrize("cols", [192, 384, 576])
def test_mma_swizzle_nonpow2_columns(cols):
    # A non-pow2 column count (e.g. head_dim 192 = 64*3) tiles the swizzle into
    # an outer mode of non-pow2 extent. The decoder must still find the swizzle,
    # which lives in the addressable low pow2 prefix of the column dim -- probing
    # only pow2-extent dims would miss it (regression: gqa_bwd TMA copy, PR #2380).
    buf = tirx.decl_buffer((128, cols), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(make_mma_swizzle_layout(buf), buf)
    assert mode is not None and _swizzle(mode) == (3, 4, 3)


def test_mma_swizzle_smooth_is_linear():
    # is_smooth=True yields an identity layout, which is linear (not swizzled).
    buf = tirx.decl_buffer((16, 64), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(make_mma_swizzle_layout(buf, is_smooth=True), buf)
    assert mode is not None and not mode.swizzle.is_swizzled


def test_core_is_dtype_agnostic_and_buffer_recasts():
    # The no-buffer core reports m_base in element-offset bits (3 for float16's
    # vector_size 8); the buffer overload recasts it to byte-space 4.
    buf = tirx.decl_buffer((8, 64), "float16", name="A", scope="shared")
    layout = make_full_bank_swizzled_layout(buf)
    assert _swizzle(cute.ComposedLayout.from_tilelang(layout)) == (3, 3, 3)
    assert _swizzle(cute.ComposedLayout.from_tilelang(layout, buf)) == (3, 4, 3)


@pytest.mark.parametrize("OFFSET", [4096, 5 * 4096, 8192 + 4096])
def test_swizzle_with_nonzero_base_offset(OFFSET):
    # A swizzle over a high, disjoint base offset: the base is carried as the
    # ComposedLayout offset (offset = Sw(A(0))), not baked into the strides.
    mode = cute.ComposedLayout.from_tilelang(_build_swizzled_layout((8, 64), lambda i, j: OFFSET + i * 64 + j, 3, 3, 3))
    assert mode is not None and _swizzle(mode) == (3, 3, 3)
    assert mode.offset == OFFSET


def test_decoder_nonpow2_quotient_keeps_linear_layout():
    # Splitting a 768-wide dim at 256 gives the version dim stride 768 =
    # 256 + 512, a two-bit column. The probe past the non-pow2 quotient
    # (j = 256) lands on bit 8 of the serialization too, and must not vouch
    # it as an identity image, or the plain version stride would masquerade
    # as a swizzle (regression: three-stage wide-linear WS kernels).
    L = Layout((3, 768), lambda v, j: [v, j // 256, j % 256])
    mode = cute.ComposedLayout.from_tilelang(L)
    assert mode is not None and not mode.swizzle.is_swizzled


def test_decoder_observes_swizzle_through_odd_extent():
    # An odd extent (5) has no pow2 sub-mode, but its in-range one-hot probes
    # (1, 2, 4) still expose the XOR source bits living in that dim: the exact
    # Sw<2,3,3> over (5,2,64) sources bit 7 from the odd dim and is recovered.
    mode = cute.ComposedLayout.from_tilelang(_build_swizzled_layout((5, 2, 64), lambda a, b, j: a * 128 + b * 64 + j, 2, 3, 3))
    assert mode is not None and _swizzle(mode) == (2, 3, 3)


def test_decoder_least_width():
    # (4,64) under an exact Sw<2,3,3> keeps every wider source bit beyond its
    # 256-element domain: the observed recovery is the minimal width, and
    # raising the least width to 3 proves (and preserves the plain layout).
    # A least width at or below the observed one is a no-op; on (8,64) bit 8
    # is a real, unswizzled row bit, so the wider pattern contradicts the
    # layout.
    small = _build_swizzled_layout((4, 64), lambda i, j: i * 64 + j, 2, 3, 3)
    observed = cute.ComposedLayout.from_tilelang(small)
    assert _swizzle(observed) == (2, 3, 3)
    widened = cute.ComposedLayout.from_tilelang(small, least_b_bits=3)
    assert widened is not None and _swizzle(widened) == (3, 3, 3)
    assert tvm.ir.structural_equal(widened.layout, observed.layout)
    assert _swizzle(cute.ComposedLayout.from_tilelang(small, least_b_bits=1)) == (2, 3, 3)
    big = _build_swizzled_layout((8, 64), lambda i, j: i * 64 + j, 2, 3, 3)
    assert cute.ComposedLayout.from_tilelang(big, least_b_bits=3) is None


@pytest.mark.parametrize(
    "b_bits,m_base,s_shift",
    [(b, m, s) for b in (1, 2, 3) for m in (0, 2, 4) for s in (1, 3, 4) if s >= b],
)
def test_synthetic_swizzle_sweep(b_bits, m_base, s_shift):
    # Recovery is exact across many (b, m, s) on a clean row-major intermediate.
    # Only s >= b (non-overlapping source/target), as in real swizzles.
    W = m_base + s_shift + b_bits + 1
    n_rows = 1 << (W - 3) if W > 3 else 1
    mode = cute.ComposedLayout.from_tilelang(_build_swizzled_layout((n_rows, 8), lambda i, j: i * 8 + j, b_bits, m_base, s_shift))
    assert mode is not None and _swizzle(mode) == (b_bits, m_base, s_shift)


@pytest.mark.parametrize(
    "shape,fwd",
    [
        ((16, 128), lambda i, j: i * 128 + j),  # row-major linear
        ((8, 8), lambda i, j: j * 8 + i),  # permutation (no XOR)
        ((3, 8), lambda i, j: i * 8 + j),  # non-pow2 leading dim, linear
    ],
)
def test_linear_layouts_detected_as_unswizzled(shape, fwd):
    # Linear / permutation layouts are recovered with b_bits == 0 (distinct from
    # the not-detectable None case).
    mode = cute.ComposedLayout.from_tilelang(Layout(shape, fwd))
    assert mode is not None and _swizzle(mode) == (0, 0, 0)


def test_general_pow2_dim_with_nonpow2_high_stride():
    # A pow2-SIZE dim may carry a non-pow2 STRIDE in high bits that is NOT a
    # swizzle source; the detector isolates the inner swizzle and keeps the high
    # strides in the residual layout.
    TILE = 8 * 64

    def addr(a, b, i, j):
        inner = (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64
        return a * (3 * TILE) + b * TILE + inner

    mode = cute.ComposedLayout.from_tilelang(Layout((2, 3, 8, 64), addr))
    assert mode is not None and _swizzle(mode) == (3, 3, 3)
    strides = list(mode.layout.stride)
    assert 1536 in strides and 512 in strides  # outer high strides untouched


def test_nonpow2_leading_size_with_swizzled_inner_tile():
    # A non-pow2 leading SIZE dim is a pass-through; the inner swizzle is still
    # recovered and the batch dim becomes a residual atom.
    def addr(batch, i, j):
        return batch * (8 * 64) + (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64

    mode = cute.ComposedLayout.from_tilelang(Layout((5, 8, 64), addr))
    assert mode is not None and _swizzle(mode) == (3, 3, 3)
    assert 5 in list(mode.layout.shape) and 512 in list(mode.layout.stride)


def test_broadcast_stride_zero_is_recovered():
    # A stride-0 (broadcast) mode is a valid affine layout (8,8):(0,1), not a
    # failure: recovery returns it unswizzled rather than rejecting it.
    mode = cute.ComposedLayout.from_tilelang(Layout((8, 8), lambda i, j: j))
    assert mode is not None and _swizzle(mode) == (0, 0, 0)
    _assert_struct(mode.layout, (8, 8), (0, 1))


def test_nonlinear_not_detectable():
    # A genuinely non-affine map (i*j is not a linear function of the coords)
    # cannot be recovered as a swizzle-over-affine layout, so it returns None.
    assert cute.ComposedLayout.from_tilelang(Layout((8, 8), lambda i, j: i * j)) is None


def test_non_constant_extent_rejected():
    # A symbolic extent has no fixed-width bit map; ICHECK requires constant.
    n = tvm.tirx.Var("n", "int32")
    with pytest.raises(Exception, match="must be constant"):
        cute.ComposedLayout.from_tilelang(Layout((n, 8), lambda i, j: i * 8 + j))


def test_decoder_preserves_input_shape():
    # ComposedLayoutFromTileLang returns a layout congruent to the TileLang input
    # shape (not a flat coalesced layout); a swizzle that splits the contiguous
    # dim yields a hierarchical sub-mode.
    buf = tirx.decl_buffer((64, 512), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(make_full_bank_swizzled_layout(buf), buf)
    # Congruent to (64, 512): in byte space the 512-elem contiguous dim is 1024
    # bytes, split by the 128B swizzle atom into (128, 8).
    assert mode.layout.shape == (64, (128, 8))


# ---------------------------------------------------------------------------
# End-to-end TMA copy with an explicitly-written swizzled layout. The forward
# index gives the design's full-bank (128B) swizzle address:
#   (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64 + (j // 64) * 4096
# ToCuteComposedLayout decodes it to Sw<3,4,3> and LowerBulk drives a swizzled
# TMA load; the box truncates at the first non-contiguous global mode so the
# rest replays as an 8-iteration tma_load loop.
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_load_with_explicit_swizzled_layout():
    import torch
    import tilelang
    import tilelang.language as T

    M, N = 64, 512

    def swizzled_addr(i, j):
        return (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64 + (j // 64) * 4096

    @T.prim_func
    def copy_swizzled(X: T.Tensor((M, N), "float16"), ok: T.Tensor((M, N), "int32")):
        with T.Kernel(1, threads=128) as _:
            S = T.alloc_shared((M, N), "float16")
            T.annotate_layout({S: Layout((M, N), swizzled_addr)})
            T.copy(X, S, prefer_instruction="tma")
            # Reading S[i, j] must return the value originally at X[i, j], since
            # annotate_layout changes only physical storage, not the logical map.
            for i, j in T.Parallel(M, N):
                ok[i, j] = T.if_then_else(S[i, j] == T.cast((i * N + j) % 2048, "float16"), 1, 0)

    buf = tirx.decl_buffer((M, N), "float16", name="S", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(Layout((M, N), swizzled_addr), buf)
    assert mode is not None and _swizzle(mode) == (3, 4, 3)

    kernel = tilelang.compile(copy_swizzled, out_idx=[1])
    src = kernel.get_kernel_source()
    assert re.search(r"for \(int \w+ = 0; \w+ < 8; \+\+\w+\) \{\n\s*tl::tma_load\(", src), "expected an 8-iteration TMA load loop"

    ii = torch.arange(M, device="cuda").view(M, 1)
    jj = torch.arange(N, device="cuda").view(1, N)
    X = ((ii * N + jj) % 2048).to(torch.float16)
    ok = kernel(X)
    assert int(ok.min()) == 1, f"{(ok == 0).sum().item()} shared elements mismatched"


# ---------------------------------------------------------------------------
# End-to-end TMA load through a position-dependent swizzle: the XOR phase
# advances across the two column blocks, so the second box (at half the
# pattern period) must land phase-shifted — which the hardware does for free.
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_load_with_position_dependent_swizzle():
    import torch
    import tilelang
    import tilelang.language as T

    M, N = 4, 64

    def posdep_halfbank(i, j):
        c = (j // 8) % 4
        s = 2 * (j // 32) + i // 2
        return (j // 32) * 128 + i * 32 + (j % 8) + (c ^ s) * 8

    @T.prim_func
    def copy_posdep(X: T.Tensor((M, N), "float16"), ok: T.Tensor((M, N), "int32")):
        with T.Kernel(1, threads=128) as _:
            S = T.alloc_shared((M, N), "float16")
            T.annotate_layout({S: Layout((M, N), posdep_halfbank)})
            T.copy(X, S, prefer_instruction="tma")
            for i, j in T.Parallel(M, N):
                ok[i, j] = T.if_then_else(S[i, j] == T.cast((i * N + j) % 2048, "float16"), 1, 0)

    kernel = tilelang.compile(copy_posdep, out_idx=[1])
    src = kernel.get_kernel_source()
    assert re.search(r"for \(int \w+ = 0; \w+ < 2; \+\+\w+\) \{\n\s*tl::tma_load\(", src)

    ii = torch.arange(M, device="cuda").view(M, 1)
    jj = torch.arange(N, device="cuda").view(1, N)
    X = ((ii * N + jj) % 2048).to(torch.float16)
    ok = kernel(X)
    assert int(ok.min()) == 1, f"{(ok == 0).sum().item()} shared elements mismatched"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_widened_full_tile_and_narrow_slice():
    """One buffer, two descriptors. (4,64) fp16 under an exact Sw<2,3,3>
    (64B span): the full-tile load carries a 128-byte contiguous run, so the
    planner widens the descriptor to the 128B mode (the extra XOR source bit
    lies beyond the buffer) and issues one box instead of an 8-step rest
    loop. The 32-column slice load is narrower than even the 64B span, which
    is legal (PTX bounds the inner box bytes by the span from above only)."""
    import torch
    import tilelang
    import tilelang.language as T

    M, N = 4, 64

    def sw233(i, j):
        addr = i * N + j
        return addr ^ ((addr & (3 << 6)) >> 3)

    @T.prim_func
    def copy_slices(X: T.Tensor((M, N), "float16"), ok: T.Tensor((M, N), "int32")):
        with T.Kernel(1, threads=128) as _:
            S = T.alloc_shared((M, N), "float16")
            T.annotate_layout({S: Layout((M, N), sw233)})
            T.copy(X, S, prefer_instruction="tma")
            T.copy(X[0, 0:32], S[1, 0:32], prefer_instruction="tma")
            for i, j in T.Parallel(M, N):
                ref = T.if_then_else((i == 1) and (j < 32), X[0, j], X[i, j])
                ok[i, j] = T.if_then_else(S[i, j] == ref, 1, 0)

    kernel = tilelang.compile(copy_slices, out_idx=[1])
    src = kernel.get_kernel_source()
    assert src.count("tl::tma_load(") == 2
    assert not re.search(r"for [^\n]*\{\n\s*tl::tma_load\(", src)

    ii = torch.arange(M, device="cuda").view(M, 1)
    jj = torch.arange(N, device="cuda").view(1, N)
    X = ((ii * N + jj) % 2048).to(torch.float16)
    ok = kernel(X)
    assert int(ok.min()) == 1, f"{(ok == 0).sum().item()} shared elements mismatched"


# ---------------------------------------------------------------------------
# Restrict: affine sub-tile of a layout, one Range per top-level mode. The
# offset is layout(mins); each mode is reshaped to its logical extent via
# with_shape, so layout(mins + c) == offset + sublayout(c). A hierarchical
# (swizzle-split) mode is flattened to the logical slice.
# ---------------------------------------------------------------------------
def test_restrict_static_slice():
    from tvm.ir import Range

    L = cute.make_layout((64, 256), (256, 1))
    off, sub = cute.restrict(L, [Range.from_min_extent(0, 64), Range.from_min_extent(64, 64)])
    assert off == 64
    assert sub.shape == (64, 64)
    assert sub.stride == (256, 1)
    # Identity at an interior coordinate.
    assert L((3, 64 + 5)) == off + sub((3, 5))


def test_restrict_dynamic_slice_origin():
    from tvm.ir import Range

    j = tvm.tirx.Var("j", "int32")
    L = cute.make_layout((64, 256), (256, 1))
    off, sub = cute.restrict(L, [Range.from_min_extent(0, 64), Range.from_min_extent(j * 64, 64)])
    ana = tvm.arith.Analyzer()
    assert ana.simplify(off - j * 64) == 0
    assert sub.shape == (64, 64)
    # Identity holds symbolically.
    assert ana.simplify(L((3, j * 64 + 5)) - (off + sub((3, 5)))) == 0


def test_restrict_rank_mismatch_rejected():
    from tvm.ir import Range

    L = cute.make_layout((64, 256), (256, 1))
    with pytest.raises(tvm.error.InternalError):
        cute.restrict(L, [Range.from_min_extent(0, 64)])  # 1 range for a rank-2 layout


def test_restrict_hierarchical_mode():
    from tvm.ir import Range

    # A hierarchical mode (a swizzle-split K dim, (64,4):(1,4096)) is addressed
    # by a single logical Range; with_shape flattens it to the j-th 64-wide tile.
    j = tvm.tirx.Var("j", "int32")
    L = cute.make_layout((64, (64, 4)), (64, (1, 4096)))
    off, sub = cute.restrict(L, [Range.from_min_extent(0, 64), Range.from_min_extent(j * 64, 64)])
    ana = tvm.arith.Analyzer()
    # Logical K-origin j*64 maps through the split layout to physical j*4096.
    assert ana.simplify(off - j * 4096) == 0
    assert sub.shape == (64, 64)
    # Identity through the hierarchical layout.
    assert ana.simplify(L((3, j * 64 + 5)) - (off + sub((3, 5)))) == 0


if __name__ == "__main__":
    tilelang.testing.main()
