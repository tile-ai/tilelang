"""InsertOOBPadding fills only the missing K tail of an MX operand."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout
from tvm import tirx
from testing.ascend._ir import calls, nodes


def _mx_operand_copy(layout, actual_k, k_alloc=128):
    mn = 32
    src_transposed = layout != "k-contiguous"
    dst_transposed = layout == "mn-contiguous"
    src_shape = (k_alloc, mn) if src_transposed else (mn, k_alloc)
    dst_shape = (k_alloc, mn) if dst_transposed else (mn, k_alloc)
    a = tirx.decl_buffer(src_shape, "float8_e4m3fn", name="A")
    l1 = tirx.decl_buffer(dst_shape, "float8_e4m3fn", name="l1", scope="shared.l1")
    scale = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="scale", scope="shared.l1")
    l0a = tirx.decl_buffer(dst_shape, "float8_e4m3fn", name="l0a", scope="shared.l0a")
    l0b = tirx.decl_buffer((mn, k_alloc), "float8_e4m3fn", name="l0b", scope="shared.l0b")
    sfa = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="sfa", scope="shared.l0a.sf")
    sfb = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="sfb", scope="shared.l0b.sf")
    acc = tirx.decl_buffer((mn, mn), "float32", name="acc", scope="shared.l0c")

    def region(buffer, transposed):
        shape = (actual_k, mn) if transposed else (mn, actual_k)
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, extent) for extent in shape])

    # The MX load and consumer identify the semantic K axis. Their other inputs
    # are already available at this pass boundary; no full kernel is needed.
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(T.copy(region(a, src_transposed), region(l1, dst_transposed), transpose=src_transposed != dst_transposed)),
            tirx.Evaluate(T.copy(l1, l0a)),
            tirx.Evaluate(T.copy(scale, sfa)),
            tirx.Evaluate(T.gemm_blockscaled(l0a, l0b, acc, sfa, sfb, transpose_A=dst_transposed, transpose_B=True, clear_accum=True)),
        ]
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[l1, scale, l0a, l0b, acc, sfa, sfb],
        annotations={"layout_map": {l1: make_ascend_major_k_layout(l1)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([a.data], tirx.SBlockRealize([], True, root), buffer_map={a.data: a})}), l1


@pytest.mark.parametrize(
    "layout, actual_k, tail",
    [
        ("k-contiguous", 65, (96, 32)),
        ("mn-contiguous", 65, (65, 63)),
        ("transpose-to-k-contiguous", 65, (96, 32)),
        ("k-contiguous", 33, None),
        ("mn-contiguous", 64, None),
        ("transpose-to-k-contiguous", 33, None),
    ],
)
def test_mx_copy_pads_only_the_unwritten_k_tail(layout, actual_k, tail):
    before, l1 = _mx_operand_copy(layout, actual_k)
    after = transform.AscendInsertOOBPadding()(before)
    fills = calls(after, "tl.tileop.fill")
    assert len(fills) == (tail is not None)
    if tail is None:
        return
    (fill,) = fills
    dst, value = fill.args
    assert dst.args[0].buffer.same_as(l1)
    assert float(value) == 0
    k_axis = 0 if layout == "mn-contiguous" else 1
    start, extent = tail
    assert [int(index) for index in dst.args[0].indices] == ([start, 0] if k_axis == 0 else [0, start])
    assert [int(size) for size in dst.args[2:]] == ([extent, 32] if k_axis == 0 else [32, extent])
    # Padding must follow the GM write and precede the MX operand's L1 read.
    copies = calls(after, "tl.tileop.ascend_copy")
    operations = nodes(after, tirx.Call)

    def position(call):
        return next(i for i, candidate in enumerate(operations) if candidate.same_as(call))

    assert position(copies[0]) < position(fill) < position(copies[1])


def test_mx_padding_rejects_an_incomplete_physical_scale_group():
    before, _ = _mx_operand_copy("k-contiguous", 96, k_alloc=96)
    with pytest.raises(tvm.error.InternalError, match="divisible by 64"):
        transform.AscendInsertOOBPadding()(before)
