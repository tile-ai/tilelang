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


def _params(mode):
    """Extract (b_bits, m_base, s_shift) from a ComposedLayout's swizzle."""
    sw = mode.swizzle
    return (sw.b_bits, sw.m_base, sw.s_shift)


def _build_swizzled_layout(shape, intermediate_fn, b_bits, m_base, s_shift):
    """Build a single-output Layout whose forward index is the swizzled
    intermediate address. `intermediate_fn(*vars)` returns the bijective
    power-of-two intermediate (pre-swizzle) physical address expression."""

    def forward(*vars):
        addr = intermediate_fn(*vars)
        # src bits = addr[m+s : m+s+b)
        src = (addr // (1 << (m_base + s_shift))) % (1 << b_bits)
        # target bits = addr[m : m+b)
        tgt = (addr // (1 << m_base)) % (1 << b_bits)
        # xor of two b-bit numbers, expressed via bit decomposition.
        x = 0
        for k in range(b_bits):
            sb = (src // (1 << k)) % 2
            tb = (tgt // (1 << k)) % 2
            x += ((sb + tb) % 2) * (1 << k)
        # clear old target bits and write swizzled bits
        return addr - tgt * (1 << m_base) + x * (1 << m_base)

    return Layout(shape, forward)


# ---------------------------------------------------------------------------
# Real bank-swizzle layouts: b_bits = 1/2/3, s_shift = 3. m_base is reported in
# byte-address space: float16 (2B) shifts the element-space base of 3 up to 4.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make_layout,expected_bbits",
    [
        (make_quarter_bank_swizzled_layout, 1),
        (make_half_bank_swizzled_layout, 2),
        (make_full_bank_swizzled_layout, 3),
    ],
)
def test_real_bank_swizzles(make_layout, expected_bbits):
    # stride=8 rows, continuous = vector_size * 2^b columns (float16 => vec=8).
    continuous = 8 * (1 << expected_bbits)
    buf = tirx.decl_buffer((8, continuous), "float16", name="A", scope="shared")
    layout = make_layout(buf)
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert mode.swizzle.is_swizzled
    # float16 => byte_shift = 1, so element-space m_base 3 -> byte-space 4.
    assert _params(mode) == (expected_bbits, 4, 3)


# The bank swizzle always acts at a fixed 16-byte (128-bit) vector granularity,
# so in byte-address space m_base is 4 regardless of dtype. The element-space
# base (log2(vector_size)) and the byte_shift (log2(bytes/elem)) compensate
# exactly: int8 (4+0), float16 (3+1), float32 (2+2) all yield 4. This is the
# invariant copy.cc relies on when it asserts m_base == 4 for every swizzle.
@pytest.mark.parametrize("dtype", ["int8", "float16", "float32"])
def test_real_bank_swizzle_mbase_is_byte_invariant(dtype):
    vector_size = 128 // int(tvm.DataType(dtype).bits)
    continuous = vector_size * 8  # full-bank (b_bits == 3) needs vec * 2^3.
    buf = tirx.decl_buffer((8, continuous), dtype, name="A", scope="shared")
    layout = make_full_bank_swizzled_layout(buf)
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert _params(mode) == (3, 4, 3)


# Expanded layouts (ExpandLayout2D) prepend pass-through leading dims whose
# extent need not be a power of two (e.g. a batch dim). The swizzle still lives
# in the trailing tile and must be detected. This is the shape from a real GEMM
# bulk copy that previously made detection return None.
@pytest.mark.parametrize("lead", [2, 3, 5])
def test_expanded_layout_with_nonpow2_leading_dim(lead):
    buf2d = tirx.decl_buffer((8, 64), "float16", name="A", scope="shared")
    layout2d = make_full_bank_swizzled_layout(buf2d)
    layout = layout2d.expand([lead])
    buf = tirx.decl_buffer((lead, 8, 64), "float16", name="A", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert _params(mode) == (3, 4, 3)


# ---------------------------------------------------------------------------
# make_mma_swizzle_layout is the widely-used MMA shared-memory swizzle. Its
# forward index uses a real T.bitwise_xor node (not arithmetic mod), exercising
# the bitwise constant-evaluation path. b_bits/s_shift scale with the row width
# (swizzle_bytes = min(128, row_bytes)); m_base is always 4 in byte-address
# space because the swizzle acts at a fixed 128-bit vector granularity.
@pytest.mark.parametrize(
    "dtype,cols,expected",
    [
        # float16 (2B): row_bytes = cols*2.
        ("float16", 32, (2, 4, 2)),  # 64B swizzle  -> b=2
        ("float16", 64, (3, 4, 3)),  # 128B swizzle -> b=3
        ("float16", 128, (3, 4, 4)),  # wider rows widen s_shift
        ("float16", 256, (3, 4, 5)),
        # float32 (4B): row_bytes = cols*4, hits 128B swizzle sooner.
        ("float32", 32, (3, 4, 3)),
        ("float32", 64, (3, 4, 4)),
        # int8 (1B).
        ("int8", 64, (2, 4, 2)),
        ("int8", 128, (3, 4, 3)),
    ],
)
def test_mma_swizzle_layout(dtype, cols, expected):
    buf = tirx.decl_buffer((16, cols), dtype, name="A", scope="shared")
    layout = make_mma_swizzle_layout(buf)
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert mode.swizzle.is_swizzled
    assert _params(mode) == expected


# The MMA swizzle's m_base is 4 in byte space regardless of dtype/width (the
# 128-bit-granularity invariant), and the leading non-pow2 row dim is handled.
@pytest.mark.parametrize("rows", [16, 64, 128])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mma_swizzle_mbase_byte_invariant(dtype, rows):
    buf = tirx.decl_buffer((rows, 64), dtype, name="A", scope="shared")
    layout = make_mma_swizzle_layout(buf)
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert mode.swizzle.m_base == 4


# is_smooth=True yields an identity layout, which is linear (not swizzled).
def test_mma_swizzle_smooth_is_linear():
    buf = tirx.decl_buffer((16, 64), "float16", name="A", scope="shared")
    layout = make_mma_swizzle_layout(buf, is_smooth=True)
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert not mode.swizzle.is_swizzled


# The dtype-agnostic core (no buffer) reports m_base in element-offset bits.
# For float16 (vector_size 8) the full-bank swizzle has element-space m_base 3;
# the buffer overload recasts it to byte-space 4.
def test_core_is_dtype_agnostic_and_buffer_recasts():
    buf = tirx.decl_buffer((8, 64), "float16", name="A", scope="shared")
    layout = make_full_bank_swizzled_layout(buf)

    core = cute.ComposedLayout.from_tilelang(layout)  # no buffer -> element space
    assert core is not None
    assert _params(core) == (3, 3, 3)

    byte_space = cute.ComposedLayout.from_tilelang(layout, buf)
    assert _params(byte_space) == (3, 4, 3)


# Recast only shifts m_base by log2(old/new); b_bits and s_shift are preserved.
# Recasting to a SMALLER element (more address bits) multiplies strides and
# shifts m_base up, and is always well-defined.
def test_recast_method():
    # Build a known swizzle in element space (b=3, m=3, s=3) via the design ex.
    layout = _build_swizzled_layout([64, 512], lambda i, j: (j % 64) + i * 64 + (j // 64) * 4096, 3, 3, 3)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert _params(mode) == (3, 3, 3)

    # m_base shifts by log2(old_bits) - log2(new_bits).
    # 16b -> 8b: smaller elements, more address bits, m_base += 1.
    assert _params(mode.recast(16, 8)) == (3, 4, 3)
    # 32b -> 8b: smaller elements, more address bits, m_base += 2.
    assert _params(mode.recast(32, 8)) == (3, 5, 3)
    # Same width is a no-op.
    assert _params(mode.recast(16, 16)) == (3, 3, 3)


# Recasting a linear (non-swizzled) mode stays linear.
def test_recast_linear_is_noop():
    mode = cute.ComposedLayout.from_tilelang(Layout([16, 128], lambda i, j: i * 128 + j))
    assert not mode.swizzle.is_swizzled
    recast = mode.recast(32, 8)
    assert not recast.swizzle.is_swizzled
    assert _params(recast) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Worked example from the design: shape [64, 512], intermediate address
# (j % 64) + i*64 + (j/64)*4096, swizzle <BBits=3, MBase=3, SShift=3>.
# Final address: (j%8) + ((i%8) ^ ((j/8)%8))*8 + i*64 + (j/64)*4096.
# ---------------------------------------------------------------------------
def test_design_worked_example():

    def intermediate(i, j):
        return (j % 64) + i * 64 + (j // 64) * 4096

    layout = _build_swizzled_layout([64, 512], intermediate, 3, 3, 3)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert _params(mode) == (3, 3, 3)


# A swizzle over a non-zero base offset (e.g. a tile base, high and disjoint
# from the swizzle bits). The address is Sw(offset + plain), so the base must be
# carried as the ComposedLayout offset and removed by unswizzling
# (offset = Sw(A(0))), rather than left baked into the recovered strides. This
# is what SetOffset in the evaluator enables; previously a non-zero base at the
# origin made recovery bail out.
@pytest.mark.parametrize("OFFSET", [4096, 5 * 4096, 8192 + 4096])
def test_swizzle_with_nonzero_base_offset(OFFSET):

    def intermediate(i, j):
        return OFFSET + i * 64 + j  # i -> source region bits, j -> low bits.

    layout = _build_swizzled_layout([8, 64], intermediate, 3, 3, 3)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert _params(mode) == (3, 3, 3)
    # The composed-layout offset is the recovered base (unswizzled), not 0.
    assert mode.offset == OFFSET


# ---------------------------------------------------------------------------
# Synthetic sweep over many (b_bits, m_base, s_shift) on a clean row-major
# intermediate layout. Verifies recovery is exact. Only s_shift >= b_bits is
# tested: overlapping source/target regions do not occur in real swizzles.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "b_bits,m_base,s_shift",
    [(b_bits, m_base, s_shift) for b_bits in (1, 2, 3) for m_base in (0, 2, 4) for s_shift in (1, 3, 4) if s_shift >= b_bits],
)
def test_synthetic_sweep(b_bits, m_base, s_shift):
    W = m_base + s_shift + b_bits + 1
    n_rows = 1 << (W - 3) if W > 3 else 1
    n_cols = 1 << 3
    # Row-major intermediate: addr = i * n_cols + j  (bijective, pow2 submodes).
    layout = _build_swizzled_layout([n_rows, n_cols], lambda i, j: i * n_cols + j, b_bits, m_base, s_shift)

    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert _params(mode) == (b_bits, m_base, s_shift)


# ---------------------------------------------------------------------------
# Linear (non-swizzled) layouts are detected as such: b_bits == 0, not None.
# This is distinct from "not detectable" (None).
# ---------------------------------------------------------------------------
def test_linear_row_major_is_linear():
    layout = Layout([16, 128], lambda i, j: i * 128 + j)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert not mode.swizzle.is_swizzled
    assert _params(mode) == (0, 0, 0)


def test_permutation_layout_is_linear():
    # Swap two bit groups: still a bijective permutation, no XOR.
    layout = Layout([8, 8], lambda i, j: j * 8 + i)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert not mode.swizzle.is_swizzled
    assert _params(mode) == (0, 0, 0)


# A non-power-of-2 leading dim is treated as a pass-through (batch) dimension,
# so a linear layout over it is still recognized as linear (not None).
def test_nonpow2_leading_dim_is_linear():
    layout = Layout([3, 8], lambda i, j: i * 8 + j)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert not mode.swizzle.is_swizzled
    assert _params(mode) == (0, 0, 0)


# ---------------------------------------------------------------------------
# General layouts: a power-of-2-SIZE dim may have a NON-power-of-2 STRIDE that
# lands in HIGH address bits and is NOT a swizzle source. The detector must
# isolate the inner swizzle from these high strides (the bug this rewrite fixes)
# and recover the outer strides into the residual CuTe layout.
# ---------------------------------------------------------------------------
def test_general_pow2_dim_with_nonpow2_high_stride():
    # Row-major [2, 3, 8, 64] over a full-bank (b=3,m=3,s=3) swizzle of the
    # inner 8x64 tile. Outer dims have non-pow2 strides (dim0: 3*8*64 = 1536,
    # dim1: 8*64 = 512) sitting above the swizzle bits.
    TILE = 8 * 64  # inner tile size in elements.

    def addr(a, b, i, j):
        # inner full-bank swizzle on (i in [0,8), j in [0,64)).
        inner = (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64
        return a * (3 * TILE) + b * TILE + inner

    layout = Layout([2, 3, 8, 64], addr)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert _params(mode) == (3, 3, 3)
    # The recovered plain layout must place the non-pow2-strided outer dims in
    # high atoms (strides 1536 and 512) untouched by the swizzle.
    strides = list(mode.layout.stride)
    assert 1536 in strides  # dim0 (size 2) high stride.
    assert 512 in strides  # dim1 (size 3) high stride.


# A non-power-of-2 leading SIZE dim combined with a swizzled inner tile: the
# leading dim is a pass-through, the inner swizzle is still recovered.
def test_nonpow2_leading_size_with_swizzled_inner_tile():
    def addr(batch, i, j):
        inner = (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64
        return batch * (8 * 64) + inner

    layout = Layout([5, 8, 64], addr)
    mode = cute.ComposedLayout.from_tilelang(layout)
    assert mode is not None
    assert _params(mode) == (3, 3, 3)
    # The size-5 batch dim becomes a residual atom of extent 5, stride 512.
    assert 5 in list(mode.layout.shape)
    assert 512 in list(mode.layout.stride)


# ---------------------------------------------------------------------------
# Not-detectable cases return None (distinct from a linear layout).
# ---------------------------------------------------------------------------
def test_non_bijective_not_detectable():
    # Collapsing layout (drops a dimension) is not injective: dim i has stride 0
    # -> None.
    layout = Layout([8, 8], lambda i, j: j)
    assert cute.ComposedLayout.from_tilelang(layout) is None


def test_non_constant_extent_not_detectable():
    # A symbolic input extent cannot be analyzed as a fixed-width bit map.
    n = tvm.tirx.Var("n", "int32")
    layout = Layout([n, 8], lambda i, j: i * 8 + j)
    assert cute.ComposedLayout.from_tilelang(layout) is None


# ---------------------------------------------------------------------------
# End-to-end TMA copy with an explicitly-written swizzled layout. The forward
# index is a single (length-1) expression giving the physical address
#
#     (j % 8) + ((i % 8) ^ ((j // 8) % 8)) * 8 + i * 64 + (j // 64) * 4096
#
# which is the design's full-bank (128B) swizzle. DetectSwizzleMode (which only
# StructuralEqual-matches the canonical maker layouts) does not recognize this
# raw form, but ToCuteComposedLayout decodes it to (b_bits=3, m_base=4,
# s_shift=3) and LowerBulk uses those parameters to drive a swizzled TMA load.
# Only the load uses TMA; the store is left as a plain copy.
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

    # Each element carries a value uniquely determined by its (i, j) coordinate,
    # exactly representable in float16 (integers in [0, 2048)). The kernel loads
    # the global tensor into a swizzle-annotated shared buffer via TMA, then
    # checks IN-KERNEL that every shared element equals the value its coordinate
    # should hold and writes the result of that comparison out. annotate_layout
    # only changes the physical storage of S, not the logical (i, j) -> value
    # mapping, so reading S[i, j] must return the value originally at X[i, j].
    @T.prim_func
    def copy_swizzled(
        X: T.Tensor((M, N), "float16"),
        ok: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(1, threads=128) as _:
            S = T.alloc_shared((M, N), "float16")
            T.annotate_layout({S: Layout([M, N], swizzled_addr)})
            # global -> shared: forced TMA load on the explicit swizzled layout.
            T.copy(X, S, prefer_instruction="tma")
            # Verify the logical mapping survived the swizzled storage: each
            # S[i, j] must equal the coordinate-determined value (i * N + j) %
            # 2048. ok[i, j] is 1 on match, 0 on mismatch.
            for i, j in T.Parallel(M, N):
                ok[i, j] = T.if_then_else(S[i, j] == T.cast((i * N + j) % 2048, "float16"), 1, 0)

    # The explicit layout decodes to the 128B full-bank swizzle.
    layout = Layout([M, N], swizzled_addr)
    buf = tirx.decl_buffer((M, N), "float16", name="S", scope="shared")
    mode = cute.ComposedLayout.from_tilelang(layout, buf)
    assert mode is not None
    assert _params(mode) == (3, 4, 3)

    # DeriveTmaTile mirrors CuTe's make_tma_copy: it truncates the descriptor box
    # at the first non-contiguous (scale != 1) global mode and replays the rest as
    # separate instructions. For this swizzle the SMEM->gmode map is
    # (64,64,8):(1@1,1@0,64@1): the box is the contiguous (64,64) (axes N,M) and
    # the trailing 8@64@axis1 becomes an 8-instruction rest (each shifts the gmem
    # N coord by 64 and the SMEM pointer by 64*64=4096). This is CuTe's exact
    # behavior -- NOT a single 3-mode box (which would mis-model per-box OOB).
    smem_plain = cute.ComposedLayout.from_tilelang(layout).layout
    gmem = cute.make_layout([M, N], stride=[N, 1])
    box, rest = cute.derive_tma_tile(gmem, smem_plain, [M, N])
    assert box == [(64, 1), (64, 0)]
    assert rest == [(8, 64, 1, 4096)]

    kernel = tilelang.compile(copy_swizzled, out_idx=[1])
    src = kernel.get_kernel_source()
    assert "tma_load" in src, "expected a TMA load for the explicit swizzled layout"
    # The box is a 2-coordinate descriptor replayed across the 8-digit rest via an
    # UNROLLED for-loop (matching the original LowerBulk's instruction-dim loop),
    # so codegen emits 8 inline tma_load( calls.
    assert src.count("tma_load(") == 8, "expected 8 (unrolled) TMA loads"

    ii = torch.arange(M, device="cuda").view(M, 1)
    jj = torch.arange(N, device="cuda").view(1, N)
    X = ((ii * N + jj) % 2048).to(torch.float16)
    ok = kernel(X)
    # Every element's in-kernel coordinate check must have passed.
    assert int(ok.min()) == 1, f"{(ok == 0).sum().item()} shared elements mismatched"


# ---------------------------------------------------------------------------
# Flat CuTe layout algebra: coalesce / inverse / composition. Each op is
# checked against a brute-force reference over the layout's whole domain.
# ---------------------------------------------------------------------------
def _size(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def _eval(layout, coord):
    """Reference CuTe idx2crd then dot-with-stride (mode 0 fastest)."""
    shape = [int(s) for s in layout.shape]
    stride = [int(s) for s in layout.stride]
    addr, rem = 0, coord
    for ext, st in zip(shape, stride):
        addr += (rem % ext) * st
        rem //= ext
    return addr


def test_cute_coalesce():
    # row-major [4,8] bit-modes collapse to a single contiguous mode.
    L = cute.make_layout([2, 2, 2, 2, 2], stride=[1, 2, 4, 8, 16])
    c = cute.coalesce(L)
    assert list(c.shape) == [32] and list(c.stride) == [1]
    # non-contiguous strides do not merge.
    L2 = cute.make_layout([8, 8], stride=[64, 1])
    c2 = cute.coalesce(L2)
    assert [list(c2.shape), list(c2.stride)] == [[8, 8], [64, 1]]
    # size-1 modes drop; full collapse yields the scalar (1, 0).
    assert list(cute.coalesce(cute.make_layout([1, 1], stride=[5, 7])).shape) == [1]


@pytest.mark.parametrize(
    "shape,stride",
    [
        ([8], [1]),
        ([4, 8], [8, 1]),  # contiguous, fastest-last
        ([8, 4], [1, 8]),  # contiguous, fastest-first
        ([2, 2, 2], [1, 2, 4]),
        ([64, 8], [1, 64]),
    ],
)
def test_cute_right_inverse(shape, stride):
    L = cute.make_layout(shape, stride=stride)
    inv = cute.right_inverse(L)
    # Defining property: L(inv(i)) == i for all i < size(inv).
    for i in range(_size([int(s) for s in inv.shape])):
        assert _eval(L, _eval(inv, i)) == i


@pytest.mark.parametrize(
    "ashape,astride,bshape,bstride",
    [
        ([8], [1], [4], [2]),
        ([4, 8], [8, 1], [8, 4], [1, 8]),
        ([12], [1], [3, 4], [4, 1]),
        ([2, 3, 4], [1, 2, 6], [6, 4], [1, 6]),
        ([64, 8], [8, 1], [8, 64], [1, 8]),
    ],
)
def test_cute_composition(ashape, astride, bshape, bstride):
    A = cute.make_layout(ashape, stride=astride)
    B = cute.make_layout(bshape, stride=bstride)
    R = cute.composition(A, B)
    # result(c) == A(B(c)) for all c in domain of B; size(result) == size(B).
    assert _size([int(s) for s in R.shape]) == _size(bshape)
    for c in range(_size(bshape)):
        assert _eval(R, c) == _eval(A, _eval(B, c))


def test_cute_right_inverse_composition_smem_to_gmem():
    # The TMA building block: composing a GMEM layout with the inverse of a
    # SMEM layout yields a SMEM-address -> GMEM-address map.
    smem = cute.make_layout([8, 64], stride=[64, 1])  # logical -> SMEM addr
    gmem = cute.make_layout([8, 64], stride=[1024, 1])  # logical -> GMEM addr (sub-tile)
    composite = cute.coalesce(cute.composition(gmem, cute.right_inverse(smem)))
    for c in range(_size([8, 64])):
        smem_addr = _eval(smem, c)
        gmem_addr = _eval(gmem, c)
        assert _eval(composite, smem_addr) == gmem_addr


def test_cute_tma_box_validity_via_composite():
    # The composite Coalesce(GMEM o RightInverse(SMEM)) is what LowerBulk uses to
    # decide TMA-expressibility: its innermost stride must be 1 (the descriptor
    # assumes globalStride[0] == 1). A row-major SMEM tile satisfies this; a
    # transposed (column-major) SMEM tile does not.
    gmem = cute.make_layout([8, 64], stride=[256, 1])  # row-major sub-tile of [8,256]

    row_major = cute.make_layout([8, 64], stride=[64, 1])
    comp_ok = cute.coalesce(cute.composition(gmem, cute.right_inverse(row_major)))
    assert list(comp_ok.stride)[0] == 1  # innermost contiguous -> TMA-valid

    transposed = cute.make_layout([8, 64], stride=[1, 8])  # i fastest in SMEM
    comp_bad = cute.coalesce(cute.composition(gmem, cute.right_inverse(transposed)))
    assert list(comp_bad.stride)[0] != 1  # innermost not contiguous -> reject


def test_derive_tma_tile_geometry():
    # DeriveTmaTile mirrors CuTe's make_tma_copy (construct_tma_gbasis) as PURE
    # layout algebra: it never permutes modes and truncates the descriptor box at
    # the first non-contiguous (scale != 1) global mode. The box covers the
    # leading unit-step run; everything past it becomes `rest` (one TMA
    # instruction per digit). Verified against a standalone CuTe ground truth.
    #
    # (a) Axis-aligned tile (standard GEMM A-tile [128,64]): the plain SMEM layout
    #     is row-major, so the whole tile is one contiguous box, no rest.
    # gmem is the global tensor layout (row-major over the tile's axes here).
    gmem2 = cute.make_layout([128, 64], stride=[64, 1])
    sp_a = cute.make_layout([128, 64], stride=[64, 1])
    box, rest = cute.derive_tma_tile(gmem2, sp_a, [128, 64])
    assert box == [(64, 1), (128, 0)]  # (extent, axis), fastest first
    assert rest == []

    # (b) Wide-swizzle split tile (B-tile [32,128]; the recovered plain SMEM
    #     layout splits the N axis as (32,64,2):(64,1,2048)). CuTe's full map is
    #     (64,32,2):(1@1,1@0,64@1) with smem_rank=2, so the box is the leading
    #     (64,32) and the trailing 2@64@axis1 becomes a 2-instruction rest mode
    #     (each digit steps gmem axis-1 by 64 and SMEM by 64*32=2048).
    gmem_b = cute.make_layout([32, 128], stride=[128, 1])
    sp_b = cute.make_layout([32, 64, 2], stride=[64, 1, 2048])
    box, rest = cute.derive_tma_tile(gmem_b, sp_b, [32, 128])
    assert box == [(64, 1), (32, 0)]
    assert rest == [(2, 64, 1, 2048)]  # (extent, scale, axis, smem_stride)

    # (c) A transposed SMEM tile: DeriveTmaTile still returns a box (it is pure
    #     layout algebra and does not see global strides), but its innermost mode
    #     reads axis 0 -- whose global stride is not 1 for a row-major tensor --
    #     so LowerBulk rejects it as non-contiguous. Confirm the innermost box
    #     mode is the non-contiguous axis (axis 0), which the global-side check in
    #     LowerBulk turns into a normal-copy fallback.
    gmem_t = cute.make_layout([8, 64], stride=[64, 1])
    sp_t = cute.make_layout([64, 8], stride=[1, 64])
    box, rest = cute.derive_tma_tile(gmem_t, sp_t, [8, 64])
    assert box[0][1] == 0  # innermost box mode reads the row axis (non-contiguous).


# A contiguous SMEM run wider than 256 elements must be split into a 256-capped
# box mode plus a `rest` iteration mode (the encoder boxDim limit is 256 and CuTe
# itself would reject >256). A [64,512] row-major tile (512 contiguous along axis
# 1) becomes box mode 256@axis1 plus rest 2@256@axis1 -- two TMA instructions,
# the second shifted 256 elements along axis 1 (SMEM stride 256).
def test_derive_tma_tile_splits_wide_inner_mode():
    gmem = cute.make_layout([64, 512], stride=[512, 1])
    sp = cute.make_layout([64, 512], stride=[512, 1])
    box, rest = cute.derive_tma_tile(gmem, sp, [64, 512])
    assert all(extent <= 256 for extent, _ in box), box
    assert box == [(256, 1), (64, 0)]
    assert rest == [(2, 256, 1, 256)]
    # Total coverage preserved: box elements * rest digits == 64 * 512.
    box_elems = 1
    for extent, _ in box:
        box_elems *= extent
    rest_digits = 1
    for extent, *_ in rest:
        rest_digits *= extent
    assert box_elems * rest_digits == 64 * 512


# DeriveTmaTile requires the plain SMEM layout to be a bijection onto
# [0, size): a gapped (padded-pitch) layout would yield a box covering only
# part of the tile. That is ICHECKed (crash, not fallback); the restriction
# can be lifted once lowering into multiple TMA instructions is supported.
def test_derive_tma_tile_rejects_gapped_layout():
    # Padded pitch: rows of 64 elements laid out every 80 (gap of 16).
    gmem = cute.make_layout([8, 64], stride=[64, 1])
    sp = cute.make_layout([64, 8], stride=[1, 80])
    with pytest.raises(Exception, match="not a bijection"):
        cute.derive_tma_tile(gmem, sp, [8, 64])


# DeriveTmaTile takes the gmem layout (CuTe construct_tma_gbasis passes the
# gtensor); its strides may be dynamic (symbolic). Composing the dynamic gmem
# with the box routes each box mode's `@axis` onto the symbolic axis stride, so
# a dynamic-shape global tensor is supported (the descriptor strides come out as
# PrimExprs). Here we just confirm derive_tma_tile succeeds with an IntExpr-stride
# gmem and the box geometry is unchanged from the static case.
def test_derive_tma_tile_dynamic_gmem():
    n = tvm.tirx.Var("n", "int32")
    gmem = cute.make_layout([128, 64], stride=[n, 1])  # row-major, dynamic N stride
    sp = cute.make_layout([128, 64], stride=[64, 1])
    box, rest = cute.derive_tma_tile(gmem, sp, [128, 64])
    assert box == [(64, 1), (128, 0)]
    assert rest == []


# ---------------------------------------------------------------------------
# Typed leaf hierarchy + full CuTe composition (incl. ScaledBasis on the RHS and
# dynamic PrimExpr strides), mirroring cute::composition_impl's is_scaled_basis
# branch and integral leaf arithmetic.
# ---------------------------------------------------------------------------
def test_make_identity_layout_leaves_are_scaled_basis():
    L = cute.make_identity_layout([4, 4])
    leaves = L.stride_leaves
    assert len(leaves) == 2
    for k, leaf in enumerate(leaves):
        assert isinstance(leaf, cute.IntTupleScaledBasis)
        assert isinstance(leaf.value, cute.IntTupleConst) and leaf.value.value == 1
        assert list(leaf.basis) == [k]


def test_int_expr_stride_roundtrips():
    s = tvm.tirx.Var("s", "int32")
    L = cute.make_layout([8], stride=[s])
    (leaf,) = L.stride_leaves
    assert isinstance(leaf, cute.IntTuplePrimExpr)
    assert tvm.ir.structural_equal(leaf.value, s)


def test_composition_rhs_scaled_basis_dynamic_strides():
    # The user's worked example: composition((16,16):(s_m,s_n), (4,4):(1@0,1@1)).
    # The RHS is an identity layout (ScaledBasis strides), routing each result
    # mode into the matching LHS axis, so the result strides are the LHS's
    # dynamic strides s_m, s_n -- exactly CuTe's is_scaled_basis(RHS) branch.
    s_m = tvm.tirx.Var("s_m", "int32")
    s_n = tvm.tirx.Var("s_n", "int32")
    A = cute.make_layout([16, 16], stride=[s_m, s_n])
    B = cute.make_identity_layout([4, 4])
    R = cute.composition(A, B)
    assert list(R.shape) == [4, 4]
    leaves = R.stride_leaves
    assert len(leaves) == 2
    assert isinstance(leaves[0], cute.IntTuplePrimExpr) and tvm.ir.structural_equal(leaves[0].value, s_m)
    assert isinstance(leaves[1], cute.IntTuplePrimExpr) and tvm.ir.structural_equal(leaves[1].value, s_n)


def test_composition_rhs_scaled_basis_const_matches_bruteforce():
    # Same RHS-basis path but with constant LHS strides, checked against the
    # defining property result(c) == A(B(c)) over the whole domain. B is the
    # identity over [4,4], so result == take the [4,4] sub-block of A.
    A = cute.make_layout([16, 16], stride=[1, 16])
    B = cute.make_identity_layout([4, 4])
    R = cute.composition(A, B)
    assert list(R.shape) == [4, 4]
    assert list(R.stride) == [1, 16]
    # The identity RHS selects the [4,4] sub-block of A: R must be the plain
    # layout [4,4]:[1,16]. Verify over the whole domain via brute force.
    ref = cute.make_layout([4, 4], stride=[1, 16])
    for c in range(_size([4, 4])):
        assert _eval(R, c) == _eval(ref, c)


# ---------------------------------------------------------------------------
# CuTeDSL-style API surface: E / ScaledBasis strides, make_layout default
# stride, layout[idx] / get / rank / size / flatten.
# ---------------------------------------------------------------------------
def test_make_layout_default_stride_is_column_major():
    # make_layout(shape) with no stride defaults to compact column-major.
    L = cute.make_layout([4, 8])
    assert list(L.shape) == [4, 8]
    assert list(L.stride) == [1, 4]


def test_E_and_scaled_basis_strides():
    # E(k) and ScaledBasis(scale, mode) build basis strides; identity == E per
    # axis.
    L = cute.make_layout([4, 4], stride=[cute.E(0), cute.E(1)])
    ident = cute.make_identity_layout([4, 4])
    for a, b in zip(L.stride_leaves, ident.stride_leaves):
        assert isinstance(a, cute.IntTupleScaledBasis)
        assert a.value.value == b.value.value and a.mode == b.mode
    # A non-unit scale: 64 * E(1).
    sb = cute.make_layout([2], stride=[cute.ScaledBasis(64, 1)]).stride_leaves[0]
    assert isinstance(sb, cute.IntTupleScaledBasis)
    assert sb.value.value == 64 and sb.mode == [1]


def test_layout_getitem_get_rank_size_flatten():
    L = cute.make_layout([4, 8], stride=[8, 1])
    assert cute.rank(L) == 2
    assert cute.size(L) == 32
    # layout[idx] and get(layout, idx) are the i-th sublayout.
    assert list(L[0].shape) == [4] and list(L[0].stride) == [8]
    assert list(cute.get(L, 1).shape) == [8] and list(cute.get(L, 1).stride) == [1]
    # flatten of an already-flat layout is itself.
    f = cute.flatten(L)
    assert list(f.shape) == [4, 8] and list(f.stride) == [8, 1]


if __name__ == "__main__":
    tilelang.testing.main()
