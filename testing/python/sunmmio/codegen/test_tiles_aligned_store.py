import pytest

from tilelang import tvm
from tilelang.layout import make_aligned_row_major, make_zz_layout
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_suvm_mlir_with_npuir_opt

# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def _build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = _to_device_kernel_func(tvm.tir.PrimFunc([], stmt))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm").inspect_source()


def _build_sunmmio_source_from_func(func):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _to_device_kernel_func(func)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm").inspect_source()


def _has_nonzero_1d_insert_slice_offset(src):
    insert_lines = [line for line in src.splitlines() if "suvm.tile.insert_slice" in line and "] [8]" in line]
    if any("[8] [8]" in line for line in insert_lines):
        return True
    return any("[%" in line for line in insert_lines) and "arith.remsi" in src


@target("Sunmmio")
def _make_nonzero_offset_aligned_store_stmt():
    bf16 = tvm.ir.PrimType("bfloat16")
    one = tvm.tir.IntImm("bool", 1)

    a_data = tvm.tir.Var("A_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    b_data = tvm.tir.Var("B_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    out_data = tvm.tir.Var("Out_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))

    a_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="A_shared", data=a_data, scope="shared.rsram")
    b_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="B_shared", data=b_data, scope="shared.rsram")
    out_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="Out_shared", data=out_data, scope="shared.rsram")

    tile_i = tvm.tir.Var("tile_i", "int32")
    ki = tvm.tir.Var("ki", "int32")
    base = tvm.tir.IntImm("int32", 8)

    value = tvm.tir.Add(
        tvm.tir.BufferLoad(a_buf, [ki + base]),
        tvm.tir.BufferLoad(b_buf, [ki + base]),
    )
    store = tvm.tir.BufferStore(out_buf, value, [ki + base])

    inner = tvm.tir.For(
        ki,
        0,
        8,
        tvm.tir.ForKind.SERIAL,
        store,
        annotations={
            "tile.interior": tvm.tir.IntImm("int32", 1),
            "tile.interior_axis": tvm.tir.IntImm("int32", 0),
        },
    )

    outer = tvm.tir.For(
        tile_i,
        0,
        1,
        tvm.tir.ForKind.SERIAL,
        inner,
        annotations={
            "tile.domain": [tvm.tir.IntImm("int32", 8)],
            "tile.execution_axis": tvm.tir.IntImm("int32", 0),
            "tile.execution_domain_axes": [tvm.tir.IntImm("int32", 0)],
            "tile.scope_entry": tvm.tir.IntImm("int32", 1),
            "tile.tile_size": [tvm.tir.IntImm("int32", 8)],
        },
    )

    return tvm.tir.DeclBuffer(
        a_buf,
        tvm.tir.DeclBuffer(
            b_buf,
            tvm.tir.DeclBuffer(
                out_buf,
                tvm.tir.Allocate(
                    a_data,
                    "bfloat16",
                    [64],
                    one,
                    tvm.tir.Allocate(
                        b_data,
                        "bfloat16",
                        [64],
                        one,
                        tvm.tir.Allocate(out_data, "bfloat16", [64], one, outer),
                    ),
                ),
            ),
        ),
    )


def _make_cross_carrier_rank1_func(dtype="bfloat16", explicit_predicate=False, read_after_store=False):
    elem_type = tvm.ir.PrimType(dtype)
    one = tvm.tir.IntImm("bool", 1)

    src_data = tvm.tir.Var("Src_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    dst_data = tvm.tir.Var("Dst_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    out_data = tvm.tir.Var("Out_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    src = tvm.tir.decl_buffer((64,), dtype, name="Src_shared", data=src_data, scope="shared.rsram")
    dst = tvm.tir.decl_buffer((64,), dtype, name="Dst_shared", data=dst_data, scope="shared.rsram")
    out = tvm.tir.decl_buffer((64,), dtype, name="Out_shared", data=out_data, scope="shared.rsram")

    segment = tvm.tir.Var("segment", "int32")
    tile_i = tvm.tir.Var("tile_i", "int32")
    lane = tvm.tir.Var("lane", "int32")
    index = segment * 14 + tile_i * 14 + lane
    predicate = lane < 13 if explicit_predicate else None
    store = tvm.tir.BufferStore(
        dst,
        tvm.tir.BufferLoad(src, [index], predicate=predicate),
        [index],
        predicate=predicate,
    )
    interior_body = store
    if read_after_store:
        interior_body = tvm.tir.SeqStmt([store, tvm.tir.BufferStore(out, tvm.tir.BufferLoad(dst, [index]), [index])])

    interior = tvm.tir.For(
        lane,
        0,
        14,
        tvm.tir.ForKind.SERIAL,
        interior_body,
        annotations={
            "tile.interior": tvm.tir.IntImm("int32", 1),
            "tile.interior_axis": tvm.tir.IntImm("int32", 0),
        },
    )
    tile_scope = tvm.tir.For(
        tile_i,
        0,
        1,
        tvm.tir.ForKind.SERIAL,
        interior,
        annotations={
            "tile.domain": [tvm.tir.IntImm("int32", 14)],
            "tile.execution_axis": tvm.tir.IntImm("int32", 0),
            "tile.execution_domain_axes": [tvm.tir.IntImm("int32", 0)],
            "tile.scope_entry": tvm.tir.IntImm("int32", 1),
            "tile.tile_size": [tvm.tir.IntImm("int32", 14)],
        },
    )
    body = tvm.tir.For(segment, 0, 4, tvm.tir.ForKind.SERIAL, tile_scope)
    if read_after_store:
        body = tvm.tir.Allocate(out_data, dtype, [64], one, body)
    body = tvm.tir.Allocate(dst_data, dtype, [64], one, body)
    body = tvm.tir.Allocate(src_data, dtype, [64], one, body)
    layout = make_aligned_row_major((64,), dtype, 64)
    layout_map = {src: layout, dst: layout}
    if read_after_store:
        body = tvm.tir.DeclBuffer(out, body)
        layout_map[out] = layout
    stmt = tvm.tir.DeclBuffer(src, tvm.tir.DeclBuffer(dst, body))
    return tvm.tir.PrimFunc([], stmt).with_attr("layout_map", layout_map)


def _make_row_major_padded_2d_aligned_store_stmt():
    bf16 = tvm.ir.PrimType("bfloat16")
    one = tvm.tir.IntImm("bool", 1)

    out_data = tvm.tir.Var("Out_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    out_buf = tvm.tir.decl_buffer((2, 40), "bfloat16", name="Out_shared", data=out_data, scope="shared.rsram")

    row = tvm.tir.Var("row", "int32")
    tile_j = tvm.tir.Var("tile_j", "int32")
    kj = tvm.tir.Var("kj", "int32")

    store = tvm.tir.BufferStore(
        out_buf,
        tvm.tir.FloatImm("bfloat16", 1.0),
        [row, tile_j * 8 + kj],
    )

    inner = tvm.tir.For(
        kj,
        0,
        8,
        tvm.tir.ForKind.SERIAL,
        store,
        annotations={
            "tile.interior": tvm.tir.IntImm("int32", 1),
            "tile.interior_axis": tvm.tir.IntImm("int32", 0),
        },
    )

    col_tiles = tvm.tir.For(
        tile_j,
        0,
        5,
        tvm.tir.ForKind.SERIAL,
        inner,
        annotations={
            "tile.execution_axis": tvm.tir.IntImm("int32", 0),
        },
    )

    rows = tvm.tir.For(
        row,
        0,
        2,
        tvm.tir.ForKind.SERIAL,
        col_tiles,
        annotations={
            "tile.domain": [tvm.tir.IntImm("int32", 2), tvm.tir.IntImm("int32", 40)],
            "tile.execution_domain_axes": [tvm.tir.IntImm("int32", 1)],
            "tile.tile_size": [tvm.tir.IntImm("int32", 8)],
        },
    )

    return tvm.tir.DeclBuffer(
        out_buf,
        tvm.tir.Allocate(out_data, "bfloat16", [2, 40], one, rows),
    )


def _make_row_major_padded_2d_aligned_store_func():
    stmt = _make_row_major_padded_2d_aligned_store_stmt()
    out_buf = stmt.buffer
    layout_map = {out_buf: make_aligned_row_major((2, 40), tvm.DataType("bfloat16"), 64)}
    return tvm.tir.PrimFunc([], stmt).with_attr("layout_map", layout_map)


def _make_small_2d_zz_carrier_func(
    tile_rows=1,
    tile_cols=1,
    dtype="float32",
    with_side_tile=False,
    domain_shape=None,
    matrix_shape=(64, 64),
    matrix_layout_kind="zz",
    explicit_predicate=False,
):
    elem_type = tvm.ir.PrimType(dtype)
    one = tvm.tir.IntImm("bool", 1)
    src_data = tvm.tir.Var("Src_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    dst_data = tvm.tir.Var("Dst_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    side_data = tvm.tir.Var("Side_shared_data", tvm.ir.PointerType(elem_type, "shared.rsram"))
    src = tvm.tir.decl_buffer(matrix_shape, dtype, name="Src_shared", data=src_data, scope="shared.rsram")
    dst = tvm.tir.decl_buffer(matrix_shape, dtype, name="Dst_shared", data=dst_data, scope="shared.rsram")
    side = tvm.tir.decl_buffer((64,), dtype, name="Side_shared", data=side_data, scope="shared.rsram")

    tile_i = tvm.tir.Var("tile_i", "int32")
    tile_j = tvm.tir.Var("tile_j", "int32")
    logical_domain = domain_shape or (tile_rows * 4, tile_cols * 4)
    tile_rows = (logical_domain[0] + 3) // 4
    tile_cols = (logical_domain[1] + 3) // 4

    def make_tile_body(suffix):
        ki = tvm.tir.Var(f"ki_{suffix}", "int32")
        kj = tvm.tir.Var(f"kj_{suffix}", "int32")
        row = tile_i * 4 + ki
        col = tile_j * 4 + kj
        predicate = tvm.tir.And(ki < 3, kj < 2) if explicit_predicate else None
        rhs = tvm.tir.BufferLoad(src, [row, col], predicate=predicate)
        rhs *= tvm.tir.BufferLoad(side, [row]) if with_side_tile else tvm.tir.FloatImm(dtype, 2.0)
        store = tvm.tir.BufferStore(dst, rhs, [row, col], predicate=predicate)
        inner = tvm.tir.For(
            kj,
            0,
            4,
            tvm.tir.ForKind.SERIAL,
            store,
            annotations={"tile.interior": 1, "tile.interior_axis": 1},
        )
        return tvm.tir.For(
            ki,
            0,
            4,
            tvm.tir.ForKind.SERIAL,
            inner,
            annotations={"tile.interior": 1, "tile.interior_axis": 0},
        )

    body = make_tile_body("full")
    if domain_shape is not None and (logical_domain[0] % 4 != 0 or logical_domain[1] % 4 != 0):
        full_tile = tvm.tir.And(tile_i * 4 + 4 <= logical_domain[0], tile_j * 4 + 4 <= logical_domain[1])
        body = tvm.tir.IfThenElse(full_tile, body, make_tile_body("tail"))
    body = tvm.tir.For(
        tile_j,
        0,
        tile_cols,
        tvm.tir.ForKind.SERIAL,
        body,
        annotations={"tile.execution_axis": 1},
    )
    body = tvm.tir.For(
        tile_i,
        0,
        tile_rows,
        tvm.tir.ForKind.SERIAL,
        body,
        annotations={
            "tile.domain": list(logical_domain),
            "tile.execution_axis": 0,
            "tile.execution_domain_axes": [0, 1],
            "tile.scope_entry": 1,
            "tile.tile_size": [4, 4],
        },
    )
    allocated_body = tvm.tir.Allocate(dst_data, dtype, list(matrix_shape), one, body)
    allocated_body = tvm.tir.Allocate(src_data, dtype, list(matrix_shape), one, allocated_body)
    if with_side_tile:
        allocated_body = tvm.tir.Allocate(side_data, dtype, [64], one, allocated_body)
    stmt = tvm.tir.DeclBuffer(
        src,
        tvm.tir.DeclBuffer(dst, tvm.tir.DeclBuffer(side, allocated_body) if with_side_tile else allocated_body),
    )
    layout = (
        make_zz_layout(matrix_shape, [0, 1], (32, 32)) if matrix_layout_kind == "zz" else make_aligned_row_major(matrix_shape, dtype, 64)
    )
    layout_map = {src: layout, dst: layout}
    if with_side_tile:
        layout_map[side] = make_aligned_row_major((64,), dtype, 64)
    return tvm.tir.PrimFunc([], stmt).with_attr("layout_map", layout_map)


def test_sunmmio_codegen_aligned_1d_store_uses_nonzero_insert_slice_offset():
    src = _build_sunmmio_source_from_stmt(_make_nonzero_offset_aligned_store_stmt())
    assert "suvm.tile.insert_slice" in src
    assert "suvm.tile.store" in src
    assert "fake_tile_store" not in src
    assert "!suvm.tile<32xbf16>" in src
    assert "!suvm.tile_view<32xbf16>" in src
    assert "!suvm.tile_view<32x1xbf16>" not in src
    assert "fake_partitioned_tile_view" not in src
    assert "fake_missing_memtensor" not in src
    assert _has_nonzero_1d_insert_slice_offset(src)


@pytest.mark.parametrize(
    "dtype,carrier,wide,mlir_dtype",
    [
        ("bfloat16", 32, 64, "bf16"),
        ("float32", 16, 32, "f32"),
    ],
)
def test_cross_carrier_rank1_load_uses_runtime_carrier_window(dtype, carrier, wide, mlir_dtype, tmp_path):
    src = _build_sunmmio_source_from_func(_make_cross_carrier_rank1_func(dtype=dtype))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=f"rank1_cross_load_{dtype}.mlir",
        opt_args=("--verify-each",),
    )
    assert f"!suvm.tile_view<{carrier}x{mlir_dtype}>" in src
    assert f"!suvm.tile<{wide}x{mlir_dtype}>" in src
    assert f"!suvm.tile_view<{wide}x{mlir_dtype}>" not in src
    assert "scf.if" in src
    assert any(
        "suvm.tile.extract_slice" in line and "[14]" in line and f"!suvm.tile<{wide}x{mlir_dtype}>" in line for line in src.splitlines()
    )


@pytest.mark.parametrize(
    "dtype,wide,mlir_dtype",
    [
        ("bfloat16", 64, "bf16"),
        ("float32", 32, "f32"),
    ],
)
def test_cross_carrier_rank1_store_uses_independent_runtime_branch(dtype, wide, mlir_dtype, tmp_path):
    src = _build_sunmmio_source_from_func(_make_cross_carrier_rank1_func(dtype=dtype))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=f"rank1_cross_store_{dtype}.mlir",
        opt_args=("--verify-each",),
    )
    assert src.count("scf.if") >= 2
    assert "suvm.tile.insert_slice" in src
    assert src.count("suvm.tile.store") >= 3
    assert f"!suvm.tile_view<{wide}x{mlir_dtype}>" not in src


@pytest.mark.parametrize(
    "dtype,mlir_dtype",
    [
        ("bfloat16", "bf16"),
        ("float32", "f32"),
    ],
)
def test_cross_carrier_rank1_predicate_selects_logical_tile(dtype, mlir_dtype, tmp_path):
    src = _build_sunmmio_source_from_func(_make_cross_carrier_rank1_func(dtype=dtype, explicit_predicate=True))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=f"rank1_cross_predicate_{dtype}.mlir",
        opt_args=("--verify-each",),
    )
    select_lines = [line for line in src.splitlines() if "suvm.tile.select" in line]
    assert select_lines
    assert all(f"!suvm.tile<14x{mlir_dtype}>" in line for line in select_lines)


def test_cross_carrier_rank1_store_invalidates_destination_value_cache(tmp_path):
    src = _build_sunmmio_source_from_func(_make_cross_carrier_rank1_func(dtype="float32", read_after_store=True))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="rank1_cross_store_then_load_float32.mlir",
        opt_args=("--verify-each",),
    )

    destination_views = set()
    saw_destination_store = False
    saw_destination_reload = False
    for line in src.splitlines():
        stripped = line.strip()
        if "suvm.get_partitioned_tile_view %1 " in stripped:
            destination_views.add(stripped.split(" =", 1)[0])
            continue
        if "suvm.tile.store" in stripped and any(f", {view} " in stripped for view in destination_views):
            saw_destination_store = True
            continue
        if saw_destination_store and "suvm.tile.load" in stripped and any(f" {view} " in stripped for view in destination_views):
            saw_destination_reload = True
            break

    assert saw_destination_store
    assert saw_destination_reload


def test_sunmmio_codegen_row_major_padded_2d_aligned_store_uses_row_block_indices():
    src = _build_sunmmio_source_from_func(_make_row_major_padded_2d_aligned_store_func())
    assert "#suvm.layout<(2, 64), (64, 1)>" in src
    aligned_view_lines = [
        line
        for line in src.splitlines()
        if "suvm.get_partitioned_tile_view" in line
        and ("tiled_dims = [1]" in line or "tiled_dims = array<i64: 1>" in line)
        and "-> !suvm.tile_view<32xbf16>" in line
    ]
    assert aligned_view_lines
    assert any("indices = [%arg0," in line for line in aligned_view_lines)
    assert "fake_partitioned_tile_view" not in src
    assert "fake_missing_memtensor" not in src


def test_sunmmio_codegen_small_2d_zz_slice_uses_register_carrier(tmp_path):
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func())
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_zz_carrier_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<4x32xf32>" in src
    assert "suvm.tile.extract_slice" in src
    assert "[4, 4]" in src
    assert "suvm.tile.mulf" in src
    assert "suvm.tile.insert_slice" in src
    assert "suvm.tile.store" in src


def test_sunmmio_codegen_small_2d_zz_carrier_uses_dynamic_slice_offset(tmp_path):
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func(tile_rows=2, tile_cols=16))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_zz_carrier_dynamic_offset_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<4x32xf32>" in src
    assert "arith.divsi" in src
    assert "arith.remsi" in src
    extract_lines = [line for line in src.splitlines() if "suvm.tile.extract_slice" in line and "[4, 4]" in line]
    insert_lines = [line for line in src.splitlines() if "suvm.tile.insert_slice" in line and "[4, 4]" in line]
    assert extract_lines
    assert insert_lines
    assert any("[%" in line for line in extract_lines)
    assert any("[%" in line for line in insert_lines)


def test_sunmmio_codegen_small_2d_bf16_zz_slice_uses_taller_carrier(tmp_path):
    dtype = "bfloat16"
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func(dtype=dtype))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=f"small_2d_{dtype}_zz_carrier_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<8x32xbf16>" in src
    assert "suvm.tile.extract_slice" in src
    assert "[4, 4]" in src
    assert "suvm.tile.insert_slice" in src


def test_sunmmio_codegen_small_2d_carrier_broadcasts_rank1_side_tile(tmp_path):
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func(with_side_tile=True))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_zz_carrier_rank1_side_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<4x32xf32>" in src
    assert "!suvm.tile_view<16xf32>" in src
    assert "!suvm.tile<4x1xf32>" in src
    assert "suvm.tile.mulf" in src
    assert "!suvm.tile<4x4xf32>, !suvm.tile<4x1xf32> -> !suvm.tile<4x4xf32>" in src
    assert "suvm.tile.insert_slice" in src


def test_sunmmio_codegen_small_2d_carrier_masks_tail_load_and_store(tmp_path):
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func(domain_shape=(6, 6)))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_zz_carrier_tail_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<4x32xf32>" in src
    assert "suvm.tile.extract_slice" in src
    assert "suvm.tile.insert_slice" in src
    assert "suvm.tile.store" in src
    logical_ops = []
    for line in src.splitlines():
        for op_name, token in (
            ("extract", "suvm.tile.extract_slice"),
            ("select", "suvm.tile.select"),
            ("mul", "suvm.tile.mulf"),
            ("insert", "suvm.tile.insert_slice"),
        ):
            if token in line:
                logical_ops.append(op_name)
                break
    assert any(logical_ops[i : i + 3] == ["extract", "select", "mul"] for i in range(len(logical_ops) - 2))
    assert any(logical_ops[i : i + 3] == ["extract", "select", "insert"] for i in range(len(logical_ops) - 2))


@pytest.mark.parametrize(
    "dtype,matrix_shape,carrier_shape,mlir_dtype",
    [
        ("float32", (64, 32), "4x32", "f32"),
        ("bfloat16", (64, 64), "4x64", "bf16"),
    ],
)
def test_sunmmio_codegen_row_major_small_2d_tile_uses_single_register_carrier(tmp_path, dtype, matrix_shape, carrier_shape, mlir_dtype):
    src = _build_sunmmio_source_from_func(
        _make_small_2d_zz_carrier_func(dtype=dtype, matrix_shape=matrix_shape, matrix_layout_kind="row_major")
    )
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_row_major_carrier_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert f"!suvm.tile_view<{carrier_shape}x{mlir_dtype}>" in src
    assert "suvm.tile.extract_slice" in src
    assert "suvm.tile.insert_slice" in src


@pytest.mark.parametrize(
    "dtype,matrix_shape",
    [
        ("float32", (64, 64)),
        ("bfloat16", (64, 128)),
    ],
)
def test_sunmmio_codegen_rejects_row_major_small_2d_tile_crossing_carrier(dtype, matrix_shape):
    with pytest.raises(
        tvm.error.InternalError,
        match="must fit entirely in one 4096-bit carrier",
    ):
        _build_sunmmio_source_from_func(
            _make_small_2d_zz_carrier_func(dtype=dtype, matrix_shape=matrix_shape, matrix_layout_kind="row_major")
        )


def test_sunmmio_codegen_small_2d_carrier_honors_explicit_load_store_predicates(tmp_path):
    src = _build_sunmmio_source_from_func(_make_small_2d_zz_carrier_func(explicit_predicate=True))
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="small_2d_zz_carrier_explicit_predicate_suvm.mlir",
        opt_args=("--verify-each",),
    )
    assert "suvm.tile.cmpi" in src
    assert "suvm.tile.andi" in src
    assert src.count("suvm.tile.select") >= 2
