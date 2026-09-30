"""AscendLowerTileOp maps logical copy regions to DMA instruction geometry."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_nz_layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_sf_layout,
)
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, nodes


def _region(buffer, extents=None, mins=None):
    return tirx.BufferRegion(
        buffer,
        [
            tvm.ir.Range.from_min_extent(m, e)
            for m, e in zip(mins if mins is not None else [0] * len(buffer.shape), extents if extents is not None else buffer.shape)
        ],
    )


def _lower(src, dst, src_region=None, dst_region=None, *, layouts=None, bindings=None, wrap=None, **kwargs):
    copy = T.copy(src_region if src_region is not None else src, dst_region if dst_region is not None else dst, **kwargs)
    buffers = [src, dst]
    for buffer in layouts or {}:
        if not any(buffer.same_as(b) for b in buffers):
            buffers.append(buffer)
    globals_ = [b for b in buffers if b.scope() == "global"]
    body = tirx.Evaluate(copy)
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        wrap(body) if wrap else body,
        alloc_buffers=[b for b in buffers if b.scope() != "global"],
        annotations={"layout_map": layouts or {}, "tl.l0_sf_bindings": bindings or {}},
    )
    target = tvm.target.Target("ascend")
    body = tirx.SBlockRealize([], True, root)
    params = [b.data for b in globals_]
    params += tirx.analysis.undefined_vars(body, params)
    func = tirx.PrimFunc(params, body, buffer_map={b.data: b for b in globals_}).with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(tvm.IRModule({"main": func}))


def _equal(actual, expected):
    assert len(actual) == len(expected)
    analyzer = tvm.arith.Analyzer()
    for a, b in zip(actual, expected):
        assert analyzer.can_prove_equal(a, b), (a, b)


@pytest.mark.parametrize(
    "src_shape,dst_shape,src_extent,dst_extent,expected",
    [
        ((16,), (16,), (16,), (16,), (1, 64, 64, 64)),
        ((16,), (4, 64), (16,), (4, 4), (4, 16, 16, 256)),
        ((4, 8), (4, 16), (4, 4), (4, 4), (4, 16, 32, 64)),
        ((4, 64), (4, 1), (4, 1), (4, 1), (4, 4, 256, 4)),
    ],
    ids=["contiguous", "split-at-ub-row", "independent-pitches", "single-element-rows"],
)
@pytest.mark.parametrize("store", [False, True], ids=["gm-to-ub", "ub-to-gm"])
def test_dma_bursts_follow_both_physical_row_boundaries(src_shape, dst_shape, src_extent, dst_extent, expected, store):
    src = tirx.decl_buffer(src_shape, "int32", name="src", scope="shared.dyn" if store else "global")
    dst = tirx.decl_buffer(dst_shape, "int32", name="dst", scope="global" if store else "shared.dyn")
    after = _lower(src, dst, _region(src, src_extent), _region(dst, dst_extent))
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_gm" if store else "tl.ascend_copy_gm_to_ubuf")
    # Burst count, bytes per burst, source pitch, destination pitch (bytes).
    indices = (3, 4, 7, 6) if store else (3, 4, 9, 10)
    _equal([copy.args[i] for i in indices], expected)


@pytest.mark.parametrize("store", [False, True], ids=["gm-to-ub", "ub-to-gm"])
@pytest.mark.parametrize("cols", [3, 4], ids=["partial-byte", "whole-byte"])
def test_packed_fp4_rows_require_whole_bytes(store, cols):
    src = tirx.decl_buffer((4, 64), "float4_e2m1fn", name="src", scope="shared.dyn" if store else "global")
    dst = tirx.decl_buffer((4, 64), "float4_e2m1fn", name="dst", scope="global" if store else "shared.dyn")
    if cols % 2:
        with pytest.raises(tvm.error.InternalError, match="packed sub-byte row must be byte-aligned"):
            _lower(src, dst, _region(src, [4, cols]), _region(dst, [4, cols]))
    else:
        after = _lower(src, dst, _region(src, [4, cols]), _region(dst, [4, cols]))
        (copy,) = calls(after, "tl.ascend_copy_ubuf_to_gm" if store else "tl.ascend_copy_gm_to_ubuf")
        _equal(copy.args[3:5], [4, 2])


def test_dynamic_rows_keep_the_stride_across_a_fixed_middle_axis():
    rows = tirx.Var("rows", "int32")
    src = tirx.decl_buffer((4, 2, 32), "float32", name="src")
    dst = tirx.decl_buffer((4, 32), "float32", name="dst", scope="shared.dyn")
    after = _lower(src, dst, _region(src, [rows, 1, 32]), _region(dst, [rows, 32]))
    (copy,) = calls(after, "tl.ascend_copy_gm_to_ubuf")
    _equal([copy.args[i] for i in (3, 4, 9, 10)], [rows, 128, 256, 128])
    # The guard belongs to instruction lowering, after scheduling sees the write.
    (guard,) = nodes(after, tirx.IfThenElse)
    for count in (0, 1, 4):
        condition = tirx.stmt_functor.substitute(guard.condition, {rows: count})
        assert bool(tvm.arith.Analyzer().simplify(condition)) == (count > 0)


def test_fixed_outer_axis_affects_the_pointer_but_not_the_l1_row_pitch():
    group = tirx.Var("group", "int32")
    src = tirx.decl_buffer((4, 16, 64), "bfloat16", name="src", strides=(2000, 80, 1))
    dst = tirx.decl_buffer((16, 64), "bfloat16", name="dst", scope="shared.l1")
    after = _lower(src, dst, _region(src, [1, 16, 64], [group, 0, 0]), layouts={dst: make_ascend_nz_layout(dst)})
    (copy,) = calls(after, "tl.ascend_copy_gm_to_cbuf")
    _equal([copy.args[i] for i in (3, 5, 6)], [160, 16, 64])
    _equal([copy.args[1].args[2]], [group * 2000])


@pytest.mark.parametrize("case", ["grouped", "tail"])
def test_fix_uses_destination_extent_and_preserves_source_pitch(case):
    rows = 16 if case == "grouped" else 128
    src = tirx.decl_buffer((rows, 16), "float32", name="src", scope="shared.l0c")
    dst = tirx.decl_buffer(
        (4, 16, 16) if case == "grouped" else (200, 16), "float32", name="dst", strides=(1000, 40, 1) if case == "grouped" else None
    )
    region = _region(dst, [1, 8, 12], [2, 0, 0]) if case == "grouped" else _region(dst, [72, 16], [128, 0])
    after = _lower(src, dst, dst_region=region, layouts={src: make_ascend_l0c_layout(src)})
    (copy,) = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    _equal(copy.args[3:7], [12, 8, 40, 16] if case == "grouped" else [16, 72, 16, 128])


@pytest.mark.parametrize("scope", ["shared.l1", "shared.dyn"])
def test_nonrepresentable_dma_geometry_is_rejected(scope):
    src = tirx.decl_buffer((2, 3, 4), "bfloat16", name="src", strides=(20, 5, 1) if scope == "shared.l1" else (14, 4, 1))
    dst = tirx.decl_buffer((6, 4) if scope == "shared.l1" else (2, 3, 5), "bfloat16", name="dst", scope=scope)
    layout = {dst: make_ascend_nz_layout(dst)} if scope == "shared.l1" else {}
    with pytest.raises(tvm.error.InternalError, match="[Oo]uter-loop lowering"):
        _lower(src, dst, dst_region=_region(dst, [6, 4] if scope == "shared.l1" else [2, 3, 4]), layouts=layout)


@pytest.mark.parametrize("symbolic", [False, True], ids=["static-region", "dynamic-region"])
@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_l1_to_l0_pitch_comes_from_the_compact_destination_region(symbolic, scope):
    m, k = (tirx.Var("m", "int32"), tirx.Var("k", "int32")) if symbolic else (17, 18)
    src = tirx.decl_buffer((64, 64), "bfloat16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((64, 64), "bfloat16", name="dst", scope=scope)
    after = _lower(
        src,
        dst,
        _region(src, [m, k]),
        _region(dst, [m, k]),
        layouts={src: make_ascend_nz_layout(src), dst: make_ascend_major_k_layout(dst)},
    )
    (copy,) = calls(after, "tl.ascend_load_cbuf_to_ca" if scope.endswith("a") else "tl.ascend_load_cbuf_to_cb")
    _equal(copy.args[4:8], [T.ceildiv(m, 16), T.ceildiv(k, 16), 4, T.ceildiv(m, 16)])


def test_pad_value_is_set_before_the_padded_dma():
    @T.prim_func
    def before(src: T.Tensor((4, 30), "float32")):
        dst = T.alloc_shared((4, 32), "float32")
        T.copy(src, dst[:, :30], pad_value=0.0)

    target = tvm.target.Target("ascend")
    with target:
        after = transform.AscendLowerTileOp()(tvm.IRModule({"main": before.with_attr("target", target)}))
    (copy,) = calls(after, "tl.ascend_copy_gm_to_ubuf")
    _equal([copy.args[i] for i in (3, 4, 5, 6, 7, 9, 10)], [4, 120, 0, 2, 1, 120, 128])
    body = after["main"].body.block.body
    assert isinstance(body, tirx.SeqStmt)
    assert body.seq[0].value.op.name == "tl.ascend_set_copy_pad_value"
    assert float(body.seq[0].value.args[0]) == 0.0
    assert body.seq[1].value.same_as(copy)


@pytest.mark.parametrize("implicit", [False, True], ids=["explicit-transpose", "layout-transpose"])
@pytest.mark.parametrize(
    "dtype,mn,k,dst_k,error",
    [
        ("bfloat16", 24, 32, 32, None),
        ("float8_e4m3fn", 64, 64, 64, None),
        ("float8_e4m3fn", 64, 48, 48, "m_step to be divisible by 2"),
        ("float8_e4m3fn", 64, 16, 32, "m_step to be divisible by 2"),
        ("float32", 32, 24, 32, None),
        ("float32", 24, 32, 32, "k_step to be divisible by 2"),
        ("float32", 32, 24, 24, "only covers 3x2"),
    ],
)
def test_transposed_load_validates_physical_groups_and_destination_capacity(implicit, dtype, mn, k, dst_k, error):
    src = tirx.decl_buffer((64, 64), dtype, name="src", scope="shared.l1")
    dst = tirx.decl_buffer((dst_k, mn) if implicit else (mn, dst_k), dtype, name="dst", scope="shared.l0a")
    layouts = {src: make_ascend_nz_layout(src), dst: make_ascend_major_mn_layout(dst) if implicit else make_ascend_major_k_layout(dst)}

    def run():
        return _lower(src, dst, _region(src, [k, mn]), layouts=layouts, transpose=not implicit)

    if error:
        with pytest.raises(tvm.error.InternalError, match=error):
            run()
    else:
        (load,) = calls(run(), "tl.ascend_load_cbuf_to_ca")
        assert int(load.args[8]) == 1


@pytest.mark.parametrize("axis,diagnostic", [(0, "source row16 origin"), (1, "source C0 origin")])
@pytest.mark.parametrize("misaligned", [False, True])
def test_alignment_checks_only_nonempty_iterations_but_preserves_the_guard(axis, diagnostic, misaligned):
    i = tirx.Var("i", "int32")
    start = T.min(i * 16, 40)
    extent = T.min(16, 40 - start)
    src = tirx.decl_buffer((64, 64), "bfloat16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((16, 16), "bfloat16", name="dst", scope="shared.l0a")
    mins, extents = [0, 0], [16, 16]
    mins[axis], extents[axis] = start + (8 if misaligned else 0), extent

    def run():
        return _lower(
            src,
            dst,
            _region(src, extents, mins),
            transpose=True,
            layouts={src: make_ascend_nz_layout(src), dst: make_ascend_major_k_layout(dst)},
            wrap=lambda body: tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body),
        )

    if misaligned:
        with pytest.raises(tvm.error.InternalError, match=diagnostic):
            run()
    else:
        after = run()
        (guard,) = nodes(after, tirx.IfThenElse)
        assert calls(guard.then_case, "tl.ascend_load_cbuf_to_ca")
        for iteration in range(4):
            condition = tirx.stmt_functor.substitute(guard.condition, {i: iteration})
            assert bool(tvm.arith.Analyzer().simplify(condition)) == (iteration < 3)


def test_simt_copy_is_lowered_per_thread_and_does_not_change_later_dma():
    src = tirx.decl_buffer((64,), "float32", name="src")
    dst = tirx.decl_buffer((64,), "float32", name="dst", scope="shared.dyn")
    # This prefix is the thread-domain marker emitted by the SimtVF frontend.
    tx = tirx.IterVar(tvm.ir.Range(0, 32), tirx.Var("simtvf_thread", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")

    def wrap(copy):
        vf = tirx.SBlock([], [], [], "SIMT_VF", tirx.AttrStmt(tx, "thread_extent", 32, copy))
        return tirx.SeqStmt([tirx.SBlockRealize([], True, vf), copy])

    after = _lower(src, dst, wrap=wrap)
    (vf,) = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "SIMT_VF"]
    assert not calls(vf, "tl.ascend_copy_gm_to_ubuf")
    assert any(load.buffer.data.same_as(src.data) for load in nodes(vf, tirx.BufferLoad))
    assert any(store.buffer.data.same_as(dst.data) for store in nodes(vf, tirx.BufferStore))
    assert len(calls(after, "tl.ascend_copy_gm_to_ubuf")) == 1


@pytest.mark.parametrize(
    "leading,index,scope,transposed,dtype,pack",
    [
        pytest.param((), (), "shared.l0a", False, "uint16", 1, id="compact-a"),
        pytest.param((3,), (1,), "shared.l0b", False, "uint16", 1, id="leading-b"),
        pytest.param((2, 3), (1, 2), "shared.l0a", False, "uint8", 2, id="two-leading-byte-scales"),
        pytest.param((2, 3), (1, 2), "shared.l0b", True, "uint16", 1, id="two-leading-transposed-b"),
        pytest.param((), (), "shared.l0a", True, "uint8", 2, id="transposed-a-byte-scales"),
    ],
)
def test_sf_copy_uses_bound_data_address_and_compact_pitch(leading, index, scope, transposed, dtype, pack):
    # No GEMM and no multi-buffer metadata: every leading dimension is ordinary.
    data = tirx.decl_buffer((*leading, *((128, 64) if transposed else (64, 128))), "float8_e4m3fn", name="data", scope=scope)
    sf = tirx.decl_buffer((*leading, 64, 2 * pack), dtype, name="sf", scope=scope + ".sf")
    src = tirx.decl_buffer((64, 4 * pack), dtype, name="source", scope="shared.l1")
    after = _lower(
        src,
        sf,
        _region(src, [32, pack], [16, pack]),
        _region(sf, [*([1] * len(leading)), 32, pack], [*index, 0, 0]),
        layouts={src: make_ascend_sf_layout(src), data: (make_ascend_major_mn_layout if transposed else make_ascend_major_k_layout)(data)},
        bindings={sf.data: data.data},
    )
    (load,) = calls(after, "tl.ascend_load_ca_sf" if scope.endswith("a") else "tl.ascend_load_cb_sf")
    tile = 0
    for size, position in zip(leading, index):
        tile = tile * size + position
    assert load.args[0].args[1].same_as(allocated_buffer(after, "data").data)
    _equal([load.args[0].args[2]], [tile * 64 * 128])
    _equal(load.args[2:], [1, 1, 2, 1, 4, 1])


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("row-origin", "source origin"),
        ("pair-origin", "source origin"),
        ("dst-origin", "zero-origin"),
        ("capacity", "bound data allocation"),
        ("compact-region", "compact destination region"),
        ("short-source", "compact destination region"),
    ],
)
def test_sf_copy_rejects_unrepresentable_physical_regions(case, diagnostic):
    src = tirx.decl_buffer((32, 8), "uint8", name="src", scope="shared.l1")
    data = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((32, 8), "uint8", name="sf", scope="shared.l0a.sf")
    src_extents = [16, 8 if case == "capacity" else 2 if case == "short-source" else 4]
    dst_extents = [16, 8 if case == "capacity" else 2 if case == "compact-region" else 4]
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _lower(
            src,
            sf,
            _region(src, src_extents, [1 if case == "row-origin" else 0, 1 if case == "pair-origin" else 0]),
            _region(sf, dst_extents, [1 if case == "dst-origin" else 0, 0]),
            layouts={src: make_ascend_sf_layout(src), data: make_ascend_major_k_layout(data)},
            bindings={sf.data: data.data},
        )


@pytest.mark.parametrize(
    "axis,lower_bound,upper_bound,valid",
    [
        pytest.param("mn", 0, 32, True, id="bounded-mn"),
        pytest.param("mn", 0, 64, False, id="unknown-mn"),
        pytest.param("mn", 33, 64, False, id="overflow-mn"),
        pytest.param("mn", 33, 32, True, id="unreachable-mn"),
        pytest.param("k", 0, 2, True, id="bounded-k"),
        pytest.param("k", 0, 8, False, id="unknown-k"),
        pytest.param("k", 3, 8, False, id="overflow-k"),
        pytest.param("k", 3, 2, True, id="unreachable-k"),
    ],
)
def test_dynamic_sf_copy_requires_proven_data_capacity(axis, lower_bound, upper_bound, valid):
    n = tirx.Var("n", "int32")
    src = tirx.decl_buffer((64, 8), "uint16", name="src", scope="shared.l1")
    sf = tirx.decl_buffer((64, 8), "uint16", name="sf", scope="shared.l0a.sf")
    data = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="data", scope="shared.l0a")
    extents = [n, 1] if axis == "mn" else [16, n]

    def lower():
        return _lower(
            src,
            sf,
            _region(src, extents),
            _region(sf, extents),
            layouts={src: make_ascend_sf_layout(src), data: make_ascend_major_k_layout(data)},
            bindings={sf.data: data.data},
            wrap=lambda body: tirx.IfThenElse(tirx.And(n >= lower_bound, n <= upper_bound), body, None),
        )

    if not valid:
        with pytest.raises(tvm.error.InternalError, match="must fit its bound data allocation"):
            lower()
        return
    (load,) = calls(lower(), "tl.ascend_load_ca_sf")
    if lower_bound <= upper_bound:
        _equal(load.args[4:6], [tirx.floordiv(n + 15, 16), 1] if axis == "mn" else [1, n])
