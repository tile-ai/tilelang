import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _make_copy_program(rows, cols, dtype, stages, versions, alias_kind):
    owner_dtype = f"uint{tvm.DataType(dtype).bits}" if alias_kind == "reinterpret" else dtype

    @T.prim_func
    def main(A: T.Tensor((4 * rows, cols), dtype)):
        with T.Kernel(1):
            if alias_kind == "direct":
                if stages == 1:
                    l1 = T.alloc_l1((rows, cols), dtype)
                else:
                    l1 = T.alloc_l1((stages, rows, cols), dtype)
            else:
                if alias_kind == "view":
                    if stages == 1:
                        owner = T.alloc_l1((rows, cols), owner_dtype)
                    else:
                        owner = T.alloc_l1((stages, rows, cols), owner_dtype)
                else:
                    owner = T.alloc_l1((stages * rows * cols,), owner_dtype)
                if stages == 1:
                    l1 = T.view(owner, (rows, cols), dtype=dtype)
                else:
                    l1 = T.view(owner, (stages, rows, cols), dtype=dtype)
            l0 = T.alloc_l0a((rows, cols), dtype)
            T.annotate_buffer_versions({l1: (versions, "iteration")})
            for i in T.Pipelined(4, num_stages=versions):
                if stages == 1:
                    T.copy(A[i * rows : (i + 1) * rows, :], l1)
                    T.copy(l1, l0)
                else:
                    T.copy(A[i * rows : (i + 1) * rows, :], l1[stages - 1, :, :])
                    T.copy(l1[stages - 1, :, :], l0)

    return main


def _lower_with_snapshots(program, auto_schedule=True):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in {
                "tl.NormalizeAscendFractalStorage",
                "tl.AscendInsertOOBPadding",
                "tl.MaterializeMultiBuffer",
                "tl.FlattenBuffer",
            }:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()], config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: auto_schedule}):
        lowered = tilelang.lower(program, target="ascend")
    return lowered, snapshots


def _l1_allocations(mod):
    buffers = []

    def collect(node):
        if isinstance(node, tirx.AllocBuffer) and node.buffer.scope() == "shared.l1":
            buffers.append(node.buffer)
        elif isinstance(node, tirx.SBlock):
            buffers.extend(buffer for buffer in node.alloc_buffers if buffer.scope() == "shared.l1")

    tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    return buffers


@pytest.mark.parametrize("rows,cols,dtype", [(15, 32, "bfloat16"), (17, 15, "bfloat16"), (16, 32, "bfloat16"), (24, 64, "float32")])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("auto_schedule", [False, True])
@pytest.mark.parametrize("alias_kind", ["direct", "reshape", "view", "reinterpret"])
def test_fractal_allocation_and_version_pitch(rows, cols, dtype, stages, auto_schedule, alias_kind):
    versions = 2 if auto_schedule else 1
    c0 = 256 // tvm.DataType(dtype).bits
    padded_rows, padded_cols = (rows + 15) // 16 * 16, (cols + c0 - 1) // c0 * c0
    tile_elements = padded_rows * padded_cols
    pitch = stages * tile_elements
    _, snapshots = _lower_with_snapshots(_make_copy_program(rows, cols, dtype, stages, versions, alias_kind), auto_schedule)
    normalized = snapshots["tl.NormalizeAscendFractalStorage"]
    assert tuple(int(dim) for dim in _l1_allocations(normalized)[0].shape) == ((stages,) if stages > 1 else ()) + (padded_rows, padded_cols)
    tvm.ir.assert_structural_equal(normalized, tilelang.ascend.transform.NormalizeAscendFractalStorage()(normalized))
    flat = snapshots["tl.FlattenBuffer"]
    assert int(_l1_allocations(flat)[0].shape[0]) == versions * pitch
    if auto_schedule:
        assert int(_l1_allocations(snapshots["tl.MaterializeMultiBuffer"])[0].strides[0]) == pitch

    offsets = []

    def collect_ptr(node):
        if (
            isinstance(node, tirx.Call)
            and node.op.name == "tirx.tvm_access_ptr"
            and node.args[1].type_annotation.storage_scope == "shared.l1"
        ):
            offsets.append(node.args[2])

    tirx.stmt_functor.post_order_visit(flat["main"].body, collect_ptr)
    assert len(offsets) == 2  # Both GM->L1 and L1->L0 use the same physical slot.
    for offset in offsets:
        variables = []
        tirx.stmt_functor.post_order_visit(
            offset, lambda node, variables=variables: variables.append(node) if isinstance(node, tirx.Var) else None
        )
        for iteration in range(4):
            value = tirx.stmt_functor.substitute(offset, {var: tirx.IntImm(var.dtype, iteration) for var in variables})
            expected = (iteration % versions) * pitch + (stages - 1) * tile_elements
            assert int(tvm.arith.Analyzer().simplify(value)) == expected


def test_fractal_alias_with_different_element_width_is_rejected():
    @T.prim_func
    def main(A: T.Tensor((15, 32), "bfloat16")):
        with T.Kernel(1):
            owner = T.alloc_l1((960,), "uint8")
            view = T.view(owner, (15, 32), dtype="bfloat16")
            T.copy(A, view)

    with pytest.raises(tvm.error.InternalError, match="Cube aliases must preserve element width"):
        _lower_with_snapshots(main)


def test_raw_pointer_does_not_silently_change_dtype():
    @T.prim_func
    def main(A: T.Tensor((15, 15), "bfloat16")):
        with T.Kernel(1):
            owner = T.alloc_l1((225,), "uint16")
            tile = T.view(owner, (15, 15), dtype="bfloat16")
            T.copy(A, tile)
            T.evaluate(T.call_extern("int32", "consume_pointer", owner.access_ptr("r", offset=15, extent=1)))

    with pytest.raises(tvm.error.InternalError, match="Cube raw pointer must use the canonical operand dtype"):
        _lower_with_snapshots(main, False)


def test_raw_fractal_pointer_uses_the_padded_row_stride():
    @T.prim_func
    def main(A: T.Tensor((15, 15), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((15, 15), "bfloat16")
            T.copy(A, l1)
            T.evaluate(T.call_extern("int32", "consume_pointer", l1.access_ptr("r", offset=15, extent=1)))

    _, snapshots = _lower_with_snapshots(main, False)
    pointers = []

    def collect(node):
        if (
            isinstance(node, tirx.Call)
            and node.op.name == "tirx.tvm_access_ptr"
            and node.args[1].type_annotation.storage_scope == "shared.l1"
        ):
            pointers.append(node)

    tirx.stmt_functor.post_order_visit(snapshots["tl.FlattenBuffer"]["main"].body, collect)
    assert int(pointers[-1].args[2]) == 16
    assert int(pointers[-1].args[3]) == 1


def test_reshaped_pointer_uses_the_original_logical_indices():
    @T.prim_func
    def main(A: T.Tensor((15, 15), "bfloat16")):
        with T.Kernel(1):
            tile = T.alloc_l1((15, 15), "bfloat16")
            alias = T.reshape(tile, (3, 75))
            T.copy(A, tile)
            for i in T.serial(3):
                T.evaluate(T.call_extern("int32", "consume_pointer", T.access_ptr(alias[i, 0], "r", extent=1)))

    _, snapshots = _lower_with_snapshots(main, False)
    pointers = []

    def collect(node):
        if isinstance(node, tirx.Call) and node.op.name == "tirx.call_extern" and node.args[0].value == "consume_pointer":
            pointers.append(node.args[1])

    tirx.stmt_functor.post_order_visit(snapshots["tl.FlattenBuffer"]["main"].body, collect)
    assert len(pointers) == 1
    offset = pointers[0].args[2]
    variables = []
    tirx.stmt_functor.post_order_visit(offset, lambda node: variables.append(node) if isinstance(node, tirx.Var) else None)
    assert len(variables) == 1
    for row in range(3):
        value = tirx.stmt_functor.substitute(offset, {variables[0]: row})
        assert int(tvm.arith.Analyzer().simplify(value)) == row * 5 * 16


def _make_alias_region_module(view_shape, mins, extents):
    owner = tirx.decl_buffer((15, 15), "bfloat16", name="owner", scope="shared.l1")
    view = tirx.decl_buffer(view_shape, "bfloat16", name="view", scope="shared.l1", data=owner.data)
    layout = tilelang.layout.make_ascend_nz_layout(owner)
    ranges = [tvm.ir.Range.from_min_extent(start, extent) for start, extent in zip(mins, extents)]
    region = tirx.BufferRegion(view, ranges)
    load = tirx.BufferLoad(view, mins)
    region_call = tirx.Call("handle", tvm.ir.Op.get("tl.region"), [load, tirx.const(1), *[tirx.const(x) for x in extents]])
    body = tirx.SeqStmt(
        [tirx.DeclBuffer(view), tirx.BufferStore(view, tirx.const(7, "bfloat16"), mins), tirx.Evaluate(load), tirx.Evaluate(region_call)]
    )
    block = tirx.SBlock(
        [],
        [region],
        [region],
        "root",
        body,
        alloc_buffers=[owner],
        annotations={"layout_map": {owner: layout, view: layout.reshape(view_shape)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})


def test_alias_indices_and_access_regions_use_the_same_padded_storage():
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((225,), (15,), (30,)))
    root = normalized["main"].body.block
    owner = root.alloc_buffers[0]
    alias = root.reads[0].buffer
    assert owner.same_as(alias)
    assert owner.data.same_as(alias.data)
    assert tuple(int(x) for x in owner.shape) == tuple(int(x) for x in alias.shape) == (16, 16)
    assert [(int(r.min), int(r.extent)) for r in root.reads[0].region] == [(1, 2), (0, 15)]
    tvm.ir.assert_structural_equal(root.reads[0], root.writes[0])
    for node in [root.body.seq[1], root.body.seq[2].value, root.body.seq[3].value.args[0]]:
        assert tuple(int(x) for x in node.indices) == (1, 0)
        assert tuple(int(x) for x in node.buffer.shape) == (16, 16)
    tvm.ir.assert_structural_equal(normalized, tilelang.ascend.transform.NormalizeAscendFractalStorage()(normalized))


def test_alias_normalization_preserves_allocation_properties():
    owner = tirx.decl_buffer((225,), "uint16", name="owner", scope="shared.l1", data_alignment=1024, offset_factor=8, elem_offset=0)
    view = tirx.decl_buffer((15, 15), "bfloat16", name="view", scope="shared.l1", data=owner.data)
    layout = tilelang.layout.make_ascend_nz_layout(view)
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.Evaluate(tirx.BufferLoad(view, [1, 0])),
        alloc_buffers=[owner],
        annotations={"layout_map": {view: layout, owner: layout.reshape((225,))}},
    )
    mod = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)
    allocation = normalized["main"].body.block.alloc_buffers[0]
    assert allocation.data.same_as(owner.data)
    assert allocation.data_alignment == 1024
    assert allocation.offset_factor == 8
    assert allocation.dtype == "bfloat16"
    assert tuple(int(x) for x in allocation.shape) == (16, 16)
    assert allocation.same_as(normalized["main"].body.block.body.value.buffer)


def test_non_rectangular_alias_region_is_rejected():
    # Three logical elements in different rows cannot become an 11-row DMA.
    with pytest.raises(tvm.error.InternalError, match="not a contiguous rectangular matrix"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 1)))


def test_overlapping_alias_region_is_rejected():
    # The 3x80 source box repeats logical elements across the 75-element row
    # stride, even though its bounding box is also 240 elements (16x15).
    with pytest.raises(tvm.error.InternalError, match="crosses an original reshape axis"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 80)))


def test_short_in_bounds_copy_keeps_its_requested_gm_region():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "bfloat16")):
        with T.Kernel(1):
            storage = T.alloc_l1((225,), "bfloat16")
            tile = T.reshape(storage, (15, 15))
            T.copy(A[:15, :15], tile)

    _, snapshots = _lower_with_snapshots(main, False)
    calls = []
    tirx.stmt_functor.post_order_visit(
        snapshots["tl.AscendInsertOOBPadding"]["main"].body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None
    )
    copy = next(node for node in calls if node.op.name == "tl.tileop.copy")
    assert [int(x) for x in copy.args[0].args[2:]] == [15, 15]
    assert [int(x) for x in copy.args[1].args[2:]] == [15, 15]
    assert not any(node.op.name == "tl.ascend_fill_l1" for node in calls)
    assert "ascend_logical_" not in str(snapshots["tl.AscendInsertOOBPadding"])


def test_larger_destination_region_alone_does_not_require_padding():
    src = tirx.decl_buffer((16, 32), "bfloat16", name="src")
    dst = tirx.decl_buffer((16, 64), "bfloat16", name="dst", scope="shared.l1")

    def region(buffer, mask):
        return tirx.Call("handle", tvm.ir.Op.get("tl.region"), [tirx.BufferLoad(buffer, [0, 0]), tirx.const(mask), *buffer.shape])

    copy = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.copy"), [region(src, 1), region(dst, 2)]))
    block = tirx.SBlock(
        [], [], [], "root", copy, alloc_buffers=[dst], annotations={"layout_map": {dst: tilelang.layout.make_ascend_nz_layout(dst)}}
    )
    mod = tvm.IRModule({"main": tirx.PrimFunc([src.data], tirx.SBlockRealize([], True, block), buffer_map={src.data: src})})
    lowered = tilelang.ascend.transform.AscendInsertOOBPadding()(mod)
    tvm.ir.assert_structural_equal(mod, lowered)


def test_padded_storage_is_the_destination_bound():
    @T.prim_func
    def main(A: T.Tensor((1, 32), "bfloat16")):
        with T.Kernel(1):
            tile = T.alloc_l1((15, 32), "bfloat16")
            T.copy(A, tile[15:16, :])

    _, snapshots = _lower_with_snapshots(main, False)
    copies = []

    def collect(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.copy":
            copies.append(node)

    tirx.stmt_functor.post_order_visit(snapshots["tl.AscendInsertOOBPadding"]["main"].body, collect)
    assert len(copies) == 1
    assert int(copies[0].args[0].args[2]) == int(copies[0].args[1].args[2]) == 1
    assert "ascend_logical_dst_shape" not in copies[0].annotations


@pytest.mark.parametrize("rows", [17, 31])
def test_column_padding_requires_the_full_logical_row_axis(rows):
    @T.prim_func
    def main(A: T.Tensor((31, 128), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((31, 64), "bfloat16")
            T.copy(A[:rows, 112:176], l1[:rows, :])

    if rows == 17:
        # 17 and 31 both round to 32. Accepting 17 would let the full-row
        # column fill overwrite valid rows 17:31 outside the requested region.
        with pytest.raises(tvm.error.InternalError, match="covering the full row axis"):
            _lower_with_snapshots(main, False)
    else:
        _lower_with_snapshots(main, False)


@pytest.mark.parametrize("cols", [17, 31])
def test_row_padding_requires_the_full_logical_col_axis(cols):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 31), "bfloat16")
            T.copy(A[8:24, :cols], l1[:, :cols])

    if cols == 17:
        with pytest.raises(tvm.error.InternalError, match="covering the full col axis"):
            _lower_with_snapshots(main, False)
    else:
        _lower_with_snapshots(main, False)


@pytest.mark.parametrize("use_alias", [False, True])
def test_full_logical_tile_padding_npu(use_alias):
    import torch

    @T.prim_func
    def main(A: T.Tensor((31, 128), "bfloat16"), B: T.Tensor((16, 64), "bfloat16"), C: T.Tensor((31, 16), "float32")):
        with T.Kernel(1):
            if use_alias:
                a_storage = T.alloc_l1((31 * 64,), "bfloat16")
                a_l1 = T.reshape(a_storage, (31, 64))
            else:
                a_l1 = T.alloc_l1((31, 64), "bfloat16")
            b_l1 = T.alloc_l1((16, 64), "bfloat16")
            a_l0 = T.alloc_l0a((31, 64), "bfloat16")
            b_l0 = T.alloc_l0b((16, 64), "bfloat16")
            if use_alias:
                c_storage = T.alloc_l0c((31, 16), "float32")
                accum = T.view(c_storage, (31, 16))
            else:
                accum = T.alloc_l0c((31, 16), "float32")
            T.copy(A[:, 112:176], a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    kernel = tilelang.compile(main, out_idx=-1)
    torch.manual_seed(42)
    a = torch.randn((31, 128), device="npu", dtype=torch.bfloat16)
    b = torch.randn((16, 64), device="npu", dtype=torch.bfloat16)
    actual = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, a[:, 112:128].float() @ b[:, :16].float().T, atol=1e-3, rtol=1e-3)


def test_short_k_slice_does_not_consume_neighboring_gm_values_npu():
    import torch

    @T.prim_func
    def main(A: T.Tensor((16, 32), "bfloat16"), B: T.Tensor((16, 32), "bfloat16"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_storage = T.alloc_l1((16 * 15,), "uint16")
            a_l1 = T.view(a_storage, (16, 15), dtype="bfloat16")
            b_l1 = T.alloc_l1((16, 15), "bfloat16")
            a_l0 = T.alloc_l0a((16, 15), "bfloat16")
            b_l0 = T.alloc_l0b((16, 15), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.copy(A[:, :15], a_l1)
            T.copy(B[:, :15], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    torch.manual_seed(42)
    a = torch.randn((16, 32), device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a[:, 15:] = 128
    b[:, 15:] = 256
    result = tilelang.compile(main, out_idx=-1)(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result, a[:, :15].float() @ b[:, :15].float().T, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
