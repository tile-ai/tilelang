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
from tilelang.intrinsics import make_mma_swizzle_layout

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
    same = cute.from_python(cute.E(0)) + cute.from_python(cute.ScaledBasis(2, 0))
    assert cute.to_python(same) == (3,)  # 1@0 + 2@0 -> (1)+(2) -> (3)
    spread = cute.from_python(cute.E(0)) + cute.from_python(cute.E(1))
    assert cute.to_python(spread) == (1, 1)  # 1@0 + 1@1 -> (1)+(0,1) -> (1,1)


def test_from_python_roundtrips_to_python():
    # from_python is the inverse of to_python for nested shapes.
    assert cute.to_python(cute.from_python([2, (3, 4)])) == (2, (3, 4))
    # A layout exposes its shape/stride as raw IntTuples for algebra.
    L = cute.make_layout((2, 3))
    assert L.shape == (2, 3)
    assert L.stride == (1, 2)


# ---------------------------------------------------------------------------
# make_layout / make_identity_layout / accessors:
# rank, size, layout[idx], layout(coord), flatten, flatten_to_tuple.
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
    (sb,) = cute.make_layout((2,), stride=(cute.ScaledBasis(64, 1),)).stride
    assert isinstance(sb, cute.ScaledBasis) and sb.value == 64 and sb.mode == (1,)
    # A dynamic (PrimExpr) stride round-trips structurally.
    s = tvm.tirx.Var("s", "int32")
    (leaf,) = cute.make_layout((8,), stride=(s,)).stride
    assert tvm.ir.structural_equal(leaf, s)


def test_rank_size_getitem_flatten():
    L = cute.make_layout((4, 8), stride=(8, 1))
    assert cute.rank(L) == 2 and cute.size(L) == 32
    # layout[idx] is the i-th sublayout (a single mode unpacks to a scalar).
    assert L[0].shape == 4 and L[0].stride == 8
    assert L[1].shape == 8 and L[1].stride == 1
    # flatten of an already-flat layout is itself.
    assert tvm.ir.structural_equal(cute.flatten(L), L)


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
    same = cute.make_layout((2, 2), stride=(cute.E(0), cute.ScaledBasis(2, 0)))
    assert all(same(c) == (c,) for c in range(4))


def test_flatten_to_tuple_collapses_hierarchy():
    # A hierarchical composition result flattens to a flat tuple of leaves.
    R = cute.composition(cute.make_layout((8, 8), stride=(8, 1)), cute.make_layout((2, 4), stride=(1, 4)))
    assert cute.flatten_to_tuple(R.shape) == tuple(int(s) for s in cute.flatten(R).shape)
    assert len(cute.flatten_to_tuple(R.shape)) == len(cute.flatten_to_tuple(R.stride))


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
    # size-1 layout collapses to the scalar (1):(0).
    _assert_struct(cute.coalesce(cute.make_layout((2, 2, 2, 2, 2), stride=(1, 2, 4, 8, 16))), (32,), (1,))
    _assert_struct(cute.coalesce(cute.make_layout((8, 8), stride=(64, 1))), (8, 8), (64, 1))
    _assert_struct(cute.coalesce(cute.make_layout((1, 1), stride=(5, 7))), (1,), (0,))
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
# Layout.from_tilelang: recover a plain affine layout (no swizzle, zero offset),
# else None.
# ---------------------------------------------------------------------------
def test_layout_from_tilelang_affine():
    L = cute.Layout.from_tilelang(Layout((16, 128), lambda i, j: i * 128 + j))
    assert L is not None
    _assert_struct(L, (16, 128), (128, 1))


def test_layout_from_tilelang_rejects_swizzle():
    swz = _build_swizzled_layout((64, 512), lambda i, j: i * 512 + j, 3, 3, 3)
    assert cute.Layout.from_tilelang(swz) is None


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


@pytest.mark.parametrize("OFFSET", [4096, 5 * 4096, 8192 + 4096])
def test_swizzle_with_nonzero_base_offset(OFFSET):
    # A swizzle over a high, disjoint base offset: the base is carried as the
    # ComposedLayout offset (offset = Sw(A(0))), not baked into the strides.
    mode = cute.ComposedLayout.from_tilelang(_build_swizzled_layout((8, 64), lambda i, j: OFFSET + i * 64 + j, 3, 3, 3))
    assert mode is not None and _swizzle(mode) == (3, 3, 3)
    assert mode.offset == OFFSET


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


# ---------------------------------------------------------------------------
# End-to-end TMA copy with an explicitly-written swizzled layout. The forward
# index gives the design's full-bank (128B) swizzle address:
#   (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64 + (j // 64) * 4096
# ToCuteComposedLayout decodes it to Sw<3,4,3> and LowerBulk drives a swizzled
# TMA load; the box truncates at the first non-contiguous global mode so the
# rest replays as 8 unrolled tma_load calls.
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
    assert src.count("tma_load(") == 8, "expected 8 (unrolled) TMA loads"

    ii = torch.arange(M, device="cuda").view(M, 1)
    jj = torch.arange(N, device="cuda").view(1, N)
    X = ((ii * N + jj) % 2048).to(torch.float16)
    ok = kernel(X)
    assert int(ok.min()) == 1, f"{(ok == 0).sum().item()} shared elements mismatched"


if __name__ == "__main__":
    tilelang.testing.main()
