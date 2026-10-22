"""InsertNd2Nz recognizes physical layouts before scheduling."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_compact_nz_layout
from tvm import tirx
from testing.ascend._ir import calls


def _module(*, rows=16, cols=128, padding=1, strides=None, staged=False, reshape=False, vf=None):
    src = tirx.decl_buffer((2, rows, cols) if staged else (rows, cols), "bfloat16", name="src", scope="shared.dyn", strides=strides)
    owner = tirx.decl_buffer((rows // 2, cols * 2), "bfloat16", name="storage", scope="shared.dyn", data=src.data) if reshape else src
    dst = tirx.decl_buffer((2, rows + padding, cols) if staged else (rows + padding, cols), "float32", name="dst", scope="shared.dyn")
    i = tirx.Var("stage", "int32")

    def region(buffer):
        ranges = [tvm.ir.Range.from_min_extent(i, 1)] if staged else []
        return tirx.BufferRegion(buffer, ranges + [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    body = tirx.Evaluate(T.copy(region(src), region(dst)))
    if vf:
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], vf, body))
    if staged:
        body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body)
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[owner, dst],
        annotations={"layout_map": {dst: make_ascend_compact_nz_layout(dst)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root))})


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
def test_scatter_regions_and_vf_boundary(case):
    before = _module(staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case))
    after = transform.InsertNd2Nz()(before)
    if case == "simt":
        tvm.ir.assert_structural_equal(after, before)
        return
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:4]] == [16, 128]
    assert [str(x.value) for x in scatter.args[5:]] == ["bfloat16_t", "float"]
    assert not calls(after, "tl.tileop.ascend_copy")
    # The destination footprint includes the bank-conflict padding row.
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(scatter.args[0].args[1])) == 16 * 128
    assert int(analyzer.simplify(scatter.args[1].args[1])) == ((128 // 8 - 1) * 17 + 16) * 8
    if case == "staged":
        src_index = scatter.args[0].args[0].indices[0]
        dst_index = scatter.args[1].args[0].indices[0]
        assert src_index.same_as(dst_index)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"padding": 0}, None),
        ({"strides": (256, 1)}, "compact trailing source matrix"),
        ({"staged": True, "strides": (2047, 128, 1)}, "32-byte-aligned source address"),
        ({"staged": True, "strides": (2048, 128, 1)}, None),
    ],
)
def test_scatter_storage_requirements(kwargs, error):
    before = _module(**kwargs)
    if error:
        with pytest.raises(ValueError, match=error):
            transform.InsertNd2Nz()(before)
    else:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_scatter")) == 1


@pytest.mark.parametrize("padding", [0, 1, 7])
@pytest.mark.parametrize("parts", [1, 2, 4, 8])
def test_partial_nz_copy_uses_complete_storage_pitches(padding, parts):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import copy, kernel, seq

    nz = tirx.decl_buffer((2, 64 + padding, 128), "bfloat16", name="nz", scope="shared.dyn")
    dst = tirx.decl_buffer((2, 64, 128), "bfloat16", name="dst", scope="shared.l1")
    rows = 64 // parts
    before = kernel(
        seq(
            *[
                copy(nz, dst, src_indices=[1, r, 16], dst_indices=[1, r, 32], src_shape=[1, rows, 64], dst_shape=[1, rows, 64])
                for r in range(0, 64, rows)
            ]
        ),
        buffers=[nz, dst],
        annotations={"layout_map": {nz: make_ascend_compact_nz_layout(nz), dst: make_ascend_nz_layout(dst)}},
    )
    after = transform.InsertNd2Nz()(before)
    copies = calls(after, "tl.ascend_nd2nz_post_copy")
    assert len(copies) == parts
    for i, call in enumerate(copies):
        assert [int(x) for x in call.args[2:6]] == [rows, 64, 64 + padding, 64]
        assert [int(x) for x in call.args[1].args[0].indices] == [1, i * rows, 16]
        assert [int(x) for x in call.args[0].args[0].indices] == [1, i * rows, 32]
        assert int(tvm.arith.Analyzer().simplify(call.args[1].args[1])) == (3 * (64 + padding) + rows) * 16


@pytest.mark.parametrize("padding", [0, 1, 7])
def test_partial_scatter_uses_destination_storage_pitch(padding):
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((16, 128), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((64 + padding, 256), "bfloat16", scope="shared.dyn")
    before = kernel(
        copy(src, dst, dst_indices=[32, 16], dst_shape=[16, 128]),
        buffers=[src, dst],
        annotations={"layout_map": {dst: make_ascend_compact_nz_layout(dst)}},
    )
    after = transform.InsertNd2Nz()(before)
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:5]] == [16, 128, 64 + padding]
    assert [int(x) for x in scatter.args[1].args[0].indices] == [32, 16]


@pytest.mark.parametrize(
    "case,error",
    [
        ("out-of-bounds", "provably in bounds"),
        ("column-start", "C0-aligned"),
        ("column-width", "C0-aligned"),
        ("dtype", "dtype conversion"),
        ("destination-layout", "MajorK destination"),
    ],
)
def test_invalid_partial_nz_regions(case, error):
    from tilelang.layout import make_ascend_nz_layout, make_ascend_major_mn_layout
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((64, 128), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((64, 128), "float16" if case == "dtype" else "bfloat16", scope="shared.l1")
    rows = 65 if case == "out-of-bounds" else 32
    cols = 63 if case == "column-width" else 64
    before = kernel(
        copy(src, dst, src_indices=[0, 1 if case == "column-start" else 0], src_shape=[rows, cols], dst_shape=[rows, cols]),
        buffers=[src, dst],
        annotations={
            "layout_map": {
                src: make_ascend_compact_nz_layout(src),
                dst: (make_ascend_major_mn_layout if case == "destination-layout" else make_ascend_nz_layout)(dst),
            }
        },
    )
    with pytest.raises(ValueError, match=error):
        transform.InsertNd2Nz()(before)


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_explicit_physical_view_pitch_does_not_depend_on_partition(parts):
    from testing.ascend._ir import kernel, seq

    # The BF16 view retains the 8192-byte FP32 owner's capacity and 64-block
    # C0-group stride without allocating additional storage.
    owner = tirx.decl_buffer((32, 64), "float32", name="owner", scope="shared.dyn")
    view = tirx.decl_buffer((4096,), "bfloat16", data=owner.data, scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    rows = 32 // parts
    before = kernel(
        seq(
            *[
                tirx.Evaluate(
                    T.ascend_nd2nz_post_copy(dst[(32 + r) * 16], view[r * 16], rows, 64, src_pitch=64, dst_pitch=64, dtype="bfloat16")
                )
                for r in range(0, 32, rows)
            ]
        ),
        buffers=[owner, dst],
    )
    after = transform.InsertNd2Nz()(before)
    copies = calls(after, "tl.ascend_nd2nz_post_copy")
    assert len(copies) == parts
    assert all([int(x) for x in call.args[2:6]] == [rows, 64, 64, 64] for call in copies)
    assert [int(x) for x in after["main"].body.block.alloc_buffers[0].shape] == [32, 64]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"src_pitch": 7}, "pitches >= rows"),
        ({"src_pitch": 65544}, "instruction field range"),
        ({"cols": 4096 * 16}, "instruction field range"),
        ({"offset": 1}, "32-byte-aligned"),
        ({"offset": 4000}, "physical span exceeds"),
    ],
)
def test_explicit_post_copy_rejects_invalid_geometry(kwargs, error):
    from testing.ascend._ir import kernel

    src = tirx.decl_buffer((4096,), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    args = {"src_pitch": 64, "dst_pitch": 64, "cols": 64, "offset": 0, **kwargs}
    call = T.ascend_nd2nz_post_copy(dst[0], src[args.pop("offset")], 8, dtype="bfloat16", **args)
    with pytest.raises(ValueError, match=error):
        transform.InsertNd2Nz()(kernel(tirx.Evaluate(call), buffers=[src, dst]))


@pytest.mark.parametrize("rows,cols", [(0, 64), (8, 0)])
def test_empty_post_copy_does_not_compute_a_negative_span(rows, cols):
    from testing.ascend._ir import kernel

    src = tirx.decl_buffer((4096,), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    before = kernel(
        tirx.Evaluate(T.ascend_nd2nz_post_copy(dst[0], src[0], rows, cols, src_pitch=64, dst_pitch=64, dtype="bfloat16")),
        buffers=[src, dst],
    )
    assert not calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")


@pytest.mark.parametrize("source_rows,rows,pitch", [(1, 1, 1), (8, 8, 8), (17, 8, 32)])
def test_canonical_nz_source_pitch_includes_fractal_rounding(source_rows, rows, pitch):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((source_rows, 64), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((32, 64), "bfloat16", scope="shared.l1")
    before = kernel(
        copy(src, dst, src_shape=[rows, 64], dst_shape=[rows, 64]),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    (call,) = calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")
    assert [int(x) for x in call.args[2:6]] == [rows, 64, pitch, 32]


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("explicit_strides", [False, True])
def test_explicit_post_copy_uses_dense_physical_layout_strides(staged, explicit_strides):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import kernel

    src_shape = (2, 65, 128) if staged else (65, 128)
    dst_shape = (2, 64, 16) if staged else (64, 16)
    src_strides = ((65 * 128, 128, 1) if staged else (128, 1)) if explicit_strides else None
    dst_strides = ((64 * 16, 16, 1) if staged else (16, 1)) if explicit_strides else None
    src = tirx.decl_buffer(src_shape, "bfloat16", scope="shared.dyn", strides=src_strides)
    dst = tirx.decl_buffer(dst_shape, "bfloat16", scope="shared.l1", strides=dst_strides)
    prefix = (1,) if staged else ()
    before = kernel(
        tirx.Evaluate(
            T.ascend_nd2nz_post_copy(dst[prefix + (0, 0)], src[prefix + (0, 16)], 64, 16, src_pitch=65, dst_pitch=64, dtype="bfloat16")
        ),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_compact_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")) == 1


@pytest.mark.parametrize("elem_offset,index,valid", [(16, 0, True), (1024, 0, True), (16, -16, False), (16, 16, False), (-16, 0, False)])
def test_explicit_post_copy_checks_capacity_relative_to_view(elem_offset, index, valid):
    from testing.ascend._ir import kernel

    owner = tirx.decl_buffer((2048,), "bfloat16", scope="shared.dyn")
    view = tirx.decl_buffer((1024,), "bfloat16", scope="shared.dyn", data=owner.data, elem_offset=elem_offset)
    dst = tirx.decl_buffer((1024,), "bfloat16", scope="shared.l1")
    before = kernel(
        tirx.Evaluate(T.ascend_nd2nz_post_copy(dst[0], view[index], 64, 16, src_pitch=64, dst_pitch=64, dtype="bfloat16")),
        buffers=[owner, dst],
    )
    if valid:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")) == 1
    else:
        with pytest.raises(ValueError, match="physical span exceeds"):
            transform.InsertNd2Nz()(before)
