"""Unit tests for CUTLASS-compatible TCGEN05 operand layouts."""

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.intrinsics.layout.mma_sm100_layout import (
    TCGEN05Meta,
    TCGEN05TmemAllocMode,
    make_tmem_frg,
    make_tmem_frg_a,
    make_tmem_frg_atom,
    make_tmem_frg_c,
    validate_tcgen05_ts_instruction,
)
from tilelang.cuda.intrinsics.macro.tcgen05_macro_generator import (
    TensorCoreIntrinEmitter,
    compute_umma_descriptor,
)
from tilelang.layout import Layout


def _swizzle_128b(shape, plain_address):
    """Construct BF16 ``Sw<3,3,3> o plain_address`` for layout probes."""

    def forward(*coords):
        address = plain_address(*coords)
        return address ^ ((address & (7 << 6)) >> 3)

    return Layout(shape, forward)


def _ranges(mins, extents):
    return [tvm.ir.Range.from_min_extent(begin, extent) for begin, extent in zip(mins, extents, strict=True)]


def _make_emitter():
    """One M256/N128/K32 BF16 emitter pinned to the (128, 64, 16) atom.

    The default row-major orientation (``a_transposed=False``,
    ``b_transposed=False``) covers both operand-under-test orientations: A is
    validated untransposed and B transposed (``not b_transposed``).
    """
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.bfloat16,
        b_dtype=T.bfloat16,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=256,
        warp_col_tiles=128,
        chunk=32,
    )
    emitter.meta = (128, 64, 16, 0, 0)
    return emitter


def _valid_operand_regions():
    """Whole-buffer A/B/C Regions aligned to ``_make_emitter``'s atom grid."""
    a_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((256, 32), "bfloat16", scope="shared"),
        _ranges((0, 0), (256, 32)),
    )
    b_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((32, 64), "bfloat16", scope="shared"),
        _ranges((0, 0), (32, 64)),
    )
    c_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((256, 128), "float32", scope="shared.tmem"),
        _ranges((0, 0), (256, 128)),
    )
    return a_region, b_region, c_region


@pytest.mark.parametrize(
    ("num_sms", "alloc_mode", "atom_m", "atom_n", "shape", "stride", "tile_stride"),
    [
        (1, TCGEN05TmemAllocMode.INTERLEAVED, 64, 32, ((16, 4), 32), ((1, 32), 128), 1),
        (1, TCGEN05TmemAllocMode.NON_INTERLEAVED, 64, 32, ((16, 4), 32), ((1, 32), 128), 2),
        (1, TCGEN05TmemAllocMode.NON_INTERLEAVED, 128, 16, (128, 16), (1, 128), 1),
        (2, TCGEN05TmemAllocMode.INTERLEAVED, 32, 32, (32, (8, 4)), (1, (128, 32)), 1),
        (2, TCGEN05TmemAllocMode.INTERLEAVED, 64, 64, (64, (32, 2)), (1, (128, 64)), 1),
        (2, TCGEN05TmemAllocMode.DUPLICATED, 64, 64, (128, 64), (1, 128), 1),
        (2, TCGEN05TmemAllocMode.INTERLEAVED, 128, 32, (128, 32), (1, 128), 1),
    ],
)
def test_tcgen05_tmem_atoms_match_cutlass(
    num_sms,
    alloc_mode,
    atom_m,
    atom_n,
    shape,
    stride,
    tile_stride,
):
    atom, actual_tile_stride = make_tmem_frg_atom(
        num_sms,
        alloc_mode,
        atom_m,
        atom_n,
    )
    assert atom.shape == shape
    assert atom.stride == stride
    assert actual_tile_stride == tile_stride


def test_tcgen05_tmem_fragment_tiles_outer_m_and_n():
    buffer = tvm.tirx.decl_buffer((256, 128), "float32", scope="shared.tmem")
    fragment = make_tmem_frg(
        buffer,
        1,
        TCGEN05TmemAllocMode.NON_INTERLEAVED,
        128,
        64,
        32,
    )
    layout = fragment.to_tilelang()
    assert [int(x) for x in layout.map_forward_index([0, 0])] == [0, 0]
    assert [int(x) for x in layout.map_forward_index([128, 0])] == [0, 64]
    assert [int(x) for x in layout.map_forward_index([0, 64])] == [0, 128]
    assert [int(x) for x in layout.map_forward_index([128, 64])] == [0, 192]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tcgen05_tmem_c_fragment_restrides_values_to_int32_storage(dtype):
    buffer = tvm.tirx.decl_buffer((128, 128), dtype, scope="shared.tmem")
    fragment = make_tmem_frg_c(buffer, TCGEN05Meta(128, 64, 16, False, False), is_ts=True)
    layout = fragment.to_tilelang()

    # CUTLASS FrgTypeC uses StorageType=int32 even when ValueType is 16-bit.
    assert [int(x) for x in layout.map_forward_index([0, 1])] == [0, 2]
    second_n_atom = [int(x) for x in layout.map_forward_index([0, 64])]
    assert second_n_atom == [0, 128]
    assert second_n_atom[1] * 16 // 32 == 64
    assert [int(x) for x in layout.get_output_shape()] == [128, 255]


def test_tcgen05_tmem_fragment_tiles_leading_batch_modes():
    buffer = tvm.tirx.decl_buffer((2, 2, 128, 64), "float32", scope="shared.tmem")
    fragment = make_tmem_frg_c(buffer, TCGEN05Meta(128, 64, 16, False, False), is_ts=True)
    layout = fragment.to_tilelang()

    # Batch modes continue the col-major tiling after the matrix tiles, with
    # the last batch axis repeating fastest.
    assert [int(x) for x in layout.map_forward_index([0, 0, 0, 0])] == [0, 0]
    assert [int(x) for x in layout.map_forward_index([0, 1, 0, 0])] == [0, 64]
    assert [int(x) for x in layout.map_forward_index([1, 1, 0, 0])] == [0, 192]
    assert [int(x) for x in layout.get_output_shape()] == [128, 256]


def test_tcgen05_tmem_fragment_rejects_unrepresentable_2sm_duplication():
    buffer = tvm.tirx.decl_buffer((64, 128), "bfloat16", scope="shared.tmem")
    with pytest.raises(ValueError, match=r"duplicated 2SM M64 allocation"):
        make_tmem_frg_a(buffer, TCGEN05Meta(128, 64, 16, False, True))


def test_tcgen05_tmem_m64_dense_fragment_interleaves_half_datapaths():
    # tmem_frg 1SM M64: the ((16,4),N):((1,32),128) atom spreads consecutive
    # M through datapath quarters; INTERLEAVED tiles the second M atom into
    # the odd 16-datapath slots of the same columns.
    buffer = tvm.tirx.decl_buffer((128, 32), "float32", scope="shared.tmem")
    fragment = make_tmem_frg_c(buffer, TCGEN05Meta(64, 32, 16, False, False), is_ts=False)
    layout = fragment.to_tilelang()
    assert [int(x) for x in layout.map_forward_index([16, 0])] == [32, 0]
    assert [int(x) for x in layout.map_forward_index([64, 0])] == [16, 0]
    assert [int(x) for x in layout.map_forward_index([127, 31])] == [127, 31]


def test_tcgen05_tmem_ws_c_fragment_matches_cutlass_ws_atom():
    # tmem_frg_ws 1SM M64: (64,(N/2,2)):(1,(128,64)) folds the upper half of N
    # into datapaths 64..127 instead of interleaving M tiles.
    buffer = tvm.tirx.decl_buffer((64, 64), "float32", scope="shared.tmem")
    fragment = make_tmem_frg_c(buffer, TCGEN05Meta(64, 64, 16, True, False), is_ts=False)
    layout = fragment.to_tilelang()
    assert [int(x) for x in layout.map_forward_index([32, 0])] == [32, 0]
    assert [int(x) for x in layout.map_forward_index([0, 32])] == [64, 0]
    assert [int(x) for x in layout.map_forward_index([63, 63])] == [127, 31]


def test_tcgen05_tmem_ws_c_fragment_requires_ws_metadata():
    buffer = tvm.tirx.decl_buffer((64, 64), "float32", scope="shared.tmem")
    # A ws=False meta selects the ordinary dense fragment, not Layout E.
    dense = make_tmem_frg_c(buffer, TCGEN05Meta(64, 64, 16, False, False), is_ts=False)
    ws = make_tmem_frg_c(buffer, TCGEN05Meta(64, 64, 16, True, False), is_ts=False)
    assert dense((0, 32)) != ws((0, 32))


@pytest.mark.parametrize(
    ("name", "shape", "dtype", "meta", "is_ts"),
    [
        # One case per PTX data-path layout variant plus the special forms.
        ("layout_a_2sm_m256", (128, 128), "float32", (256, 128, 16, 0, 1), False),
        ("layout_b_2sm_m128", (64, 128), "float32", (128, 128, 16, 0, 1), False),
        ("layout_d_1sm_m128", (128, 128), "float32", (128, 128, 16, 0, 0), False),
        ("layout_e_ws_m64", (64, 64), "float32", (64, 64, 16, 1, 0), False),
        ("layout_f_1sm_m64", (128, 32), "float32", (64, 32, 16, 0, 0), False),
        ("layout_g_ws_m32", (32, 128), "float32", (32, 128, 16, 1, 0), False),
        ("f16_int32_storage", (128, 64), "float16", (128, 64, 16, 0, 0), True),
        ("leading_batch", (2, 128, 64), "float32", (128, 64, 16, 0, 0), False),
    ],
)
def test_tcgen05_tmem_fragment_tilelang_roundtrip(name, shape, dtype, meta, is_ts):
    """to_tilelang and from_tilelang_hierarchical are exact inverses.

    The TileLang form is what travels through the layout map; recovery must
    reproduce the hierarchical ``(datapath, column)`` mapping, not a
    serialized linear address.
    """
    import itertools

    from tilelang.layout import cute

    buffer = tvm.tirx.decl_buffer(shape, dtype, scope="shared.tmem")
    fragment = make_tmem_frg_c(buffer, TCGEN05Meta.from_ffi(meta), is_ts=is_ts)
    recovered = cute.Layout.from_tilelang_hierarchical(fragment.to_tilelang())
    assert recovered is not None, f"{name} is not decodable by the CuTe analyzer"
    extents = [cute.size(fragment[mode]) for mode in range(cute.rank(fragment))]
    for coords in itertools.product(*(range(0, extent, max(1, extent // 7)) for extent in extents)):
        assert fragment(coords) == recovered(coords), f"{name} diverges at {coords}"


@pytest.mark.parametrize(
    "meta",
    [
        (64, 8, 16, 0, 0),
        (64, 256, 16, 0, 0),
        (128, 16, 16, 0, 0),
        (128, 256, 16, 0, 0),
        (128, 32, 16, 0, 1),
        (128, 256, 16, 0, 1),
        (256, 32, 16, 0, 1),
        (256, 256, 16, 0, 1),
    ],
)
def test_tcgen05_ts_instruction_legal_boundaries_match_cutlass(meta):
    validate_tcgen05_ts_instruction(TCGEN05Meta.from_ffi(meta), T.bfloat16, False)


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        ((128, 8, 16, 0, 0), r"1CTA.*N.*multiple of 16"),
        ((128, 24, 16, 0, 0), r"1CTA.*N.*multiple of 16"),
        ((128, 16, 16, 0, 1), r"2CTA.*N.*multiple of 32"),
        ((256, 48, 16, 0, 1), r"2CTA.*N.*multiple of 32"),
    ],
)
def test_tcgen05_ts_instruction_rejects_cutlass_illegal_boundaries(meta, message):
    with pytest.raises(ValueError, match=message):
        validate_tcgen05_ts_instruction(TCGEN05Meta.from_ffi(meta), T.bfloat16, False)


@pytest.mark.parametrize(
    ("operand", "shape", "mins", "extents", "transposed", "mn_atom", "k_atom", "message"),
    [
        ("A", (256, 16), (0, 0), (192, 16), False, 128, 16, r"A M extent 192.*atom 128"),
        ("A", (128, 32), (0, 0), (128, 24), False, 128, 16, r"A K extent 24.*atom 16"),
        ("B", (32, 64), (0, 0), (24, 64), True, 64, 16, r"B K extent 24.*atom 16"),
        ("B", (16, 128), (0, 0), (16, 96), True, 64, 16, r"B N extent 96.*atom 64"),
    ],
)
def test_tcgen05_matrix_regions_reject_partial_atom_extents(
    operand,
    shape,
    mins,
    extents,
    transposed,
    mn_atom,
    k_atom,
    message,
):
    emitter = _make_emitter()
    a_region, b_region, c_region = _valid_operand_regions()
    sliced = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer(shape, "bfloat16", scope="shared"),
        _ranges(mins, extents),
    )
    regions = {"A": a_region, "B": b_region, "C": c_region, operand: sliced}
    with pytest.raises(AssertionError, match=message):
        emitter.validate_tcgen05_operand_regions(regions["A"], regions["B"], regions["C"], is_ts=False)


@pytest.mark.parametrize(
    ("extents", "message"),
    [
        ((192, 64), r"C M extent 192.*atom 128"),
        ((128, 96), r"C N extent 96.*atom 64"),
    ],
)
def test_tcgen05_accumulator_regions_reject_partial_atom_extents(extents, message):
    emitter = _make_emitter()
    a_region, b_region, _ = _valid_operand_regions()
    c_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((256, 128), "float32", scope="shared.tmem"),
        [tvm.ir.Range.from_min_extent(0, extent) for extent in extents],
    )
    with pytest.raises(AssertionError, match=message):
        emitter.validate_tcgen05_operand_regions(a_region, b_region, c_region, is_ts=False)


def test_tcgen05_matrix_region_rejects_wide_leading_mode():
    emitter = _make_emitter()
    _, b_region, c_region = _valid_operand_regions()
    a_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((2, 128, 16), "bfloat16", scope="shared"),
        _ranges((0, 0, 0), (2, 128, 16)),
    )
    with pytest.raises(AssertionError, match=r"A leading mode 0 must have unit extent"):
        emitter.validate_tcgen05_operand_regions(a_region, b_region, c_region, is_ts=False)


def test_tcgen05_operand_validation_pins_leading_modes():
    emitter = _make_emitter()
    a_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((2, 3, 256, 128), "bfloat16", scope="shared"),
        _ranges((1, 2, 128, 16), (1, 1, 128, 32)),
    )
    b_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((32, 64), "bfloat16", scope="shared"),
        _ranges((0, 0), (32, 64)),
    )
    c_region = tvm.tirx.BufferRegion(
        tvm.tirx.decl_buffer((2, 3, 256, 128), "float32", scope="shared.tmem"),
        _ranges((1, 1, 128, 64), (1, 1, 128, 64)),
    )
    emitter.validate_tcgen05_operand_regions(a_region, b_region, c_region, is_ts=False)


def test_tcgen05_descriptor_pins_leading_stage_mode():
    """A pipeline-stage mode pinned by the Region shifts only the slice base."""

    m_extent = k_extent = 128

    def plain_2d(m, k):
        return m * 64 + k % 64 + (k // 64) * m_extent * 64

    reference_buffer = tvm.tirx.decl_buffer((m_extent, k_extent), "bfloat16", scope="shared")
    reference = compute_umma_descriptor(
        _swizzle_128b((m_extent, k_extent), plain_2d),
        reference_buffer,
        False,
        region=_ranges((0, 64), (m_extent, 32)),
    )

    parent_shape = (3, m_extent, k_extent)
    parent_buffer = tvm.tirx.decl_buffer(parent_shape, "bfloat16", scope="shared")
    parent_layout = _swizzle_128b(
        parent_shape,
        lambda stage, m, k: stage * m_extent * k_extent + plain_2d(m, k),
    )
    staged = compute_umma_descriptor(
        parent_layout,
        parent_buffer,
        False,
        region=_ranges((2, 0, 64), (1, m_extent, 32)),
    )

    assert (
        staged.leading_byte_offset,
        staged.stride_byte_offset,
        staged.is_k_major,
    ) == (
        reference.leading_byte_offset,
        reference.stride_byte_offset,
        reference.is_k_major,
    )
    assert staged.slice_byte_offset - reference.slice_byte_offset == 2 * m_extent * k_extent * 2


def test_tcgen05_descriptor_k_panel_stride_follows_the_buffer():
    """An MN slice keeps the buffer's K-panel spacing, not the slice's.

    ``block_K`` past one swizzle atom stores the operand as (MN, (atom, panels)),
    so stepping to the next K panel moves by ``buffer_MN * atom`` elements. That
    step must be read off the layout: reconstructing it from the operand extent
    (``mn_extent * atom``) halves the stride for a half-height slice and makes
    every K atom past the first read the wrong panel.
    """

    # Two M atoms tall so the slice origin stays on the 128-row atom grid that
    # validate_tcgen05_operand_regions requires, and two K panels wide.
    m_extent, k_extent = 256, 128
    atom_elems = 64  # 128B swizzle atom over bfloat16

    def plain_2d(m, k):
        return m * atom_elems + k % atom_elems + (k // atom_elems) * m_extent * atom_elems

    layout = _swizzle_128b((m_extent, k_extent), plain_2d)
    buffer = tvm.tirx.decl_buffer((m_extent, k_extent), "bfloat16", scope="shared")

    whole = compute_umma_descriptor(layout, buffer, False, region=_ranges((0, 0), (m_extent, k_extent)))
    sliced = compute_umma_descriptor(layout, buffer, False, region=_ranges((128, 0), (128, k_extent)))

    assert whole.k_panel_elems == m_extent * atom_elems
    # Same spacing for the slice, even though its MN extent is half as large.
    assert sliced.k_panel_elems == m_extent * atom_elems
    assert sliced.slice_byte_offset == 128 * atom_elems * 2
