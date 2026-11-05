"""Native MMA generator contracts before device-specific dispatch/lowering."""

from itertools import product
import sys

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang.language as T
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter
from tilelang.tileop._lowering import close_lowered_function


def _tensor(name, shape, dtype="float16", scope="local.fragment"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout="default"))


def _emitter(**kwargs):
    options = dict(
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,
        chunk=16,
        accum_dtype="float32",
        thread_var=ir.Var("tx", "int32"),
    )
    options.update(kwargs)
    return TensorCoreIntrinEmitter(**options)


def _integer(expr):
    return int(sym.Analyzer().simplify(expr))


def test_mma_accepts_dtype_objects_from_native_tensor_parameters():
    source = _tensor("A", (16, 16))
    output = _tensor("C", (16, 16), "float32")
    emitter = _emitter(a_dtype=source.dtype, b_dtype=source.dtype, accum_dtype=output.dtype)
    assert emitter.a_dtype_abbrv == emitter.b_dtype_abbrv == "fp16"
    assert emitter.accum_dtype_abbrv == "fp32"
    assert emitter.k_dim == 16


@pytest.mark.parametrize("row_warps,col_warps,row_tiles,col_tiles", [(1, 1, 16, 16), (1, 1, 16, 8), (2, 2, 32, 16)])
def test_mma_store_layout_matches_instruction_owner_and_slot(row_warps, col_warps, row_tiles, col_tiles):
    emitter = _emitter(block_row_warps=row_warps, block_col_warps=col_warps, warp_row_tiles=row_tiles, warp_col_tiles=col_tiles)
    shape = (row_warps * row_tiles, col_warps * col_tiles)
    tensor = _tensor("C", shape, "float32")
    layout = emitter.make_mma_store_layout(tensor)
    slots = set()
    for row, column in product(range(shape[0]), range(shape[1])):
        warp_m, warp_n = row // row_tiles, column // col_tiles
        lane = (row % 8) * 4 + (column % 8) // 2
        expected_owner = (warp_n * row_warps + warp_m) * 32 + lane
        local_m = (row % row_tiles) // 16
        local_n = (column % col_tiles) // emitter.n_dim
        atom_slot = (column % emitter.n_dim // 8) * 4 + (row % 16 // 8) * 2 + column % 2
        expected_slot = (local_m * emitter.warp_cols + local_n) * emitter.local_size_out + atom_slot
        owner = _integer(layout.map_forward_thread([row, column])[0])
        slot = _integer(layout.map_forward_index([row, column])[0])
        assert (owner, slot) == (expected_owner, expected_slot)
        slots.add((owner, slot))
    assert len(slots) == shape[0] * shape[1]
    assert "tvm" not in sys.modules


def test_mma_region_legalization_preserves_native_identity_and_slice_origin():
    source = _tensor("shared", (32, 32), scope="shared")
    ranges = [ir.Range.from_min_extent(3, 16), ir.Range.from_min_extent(5, 16)]
    region = ir.TensorRegion(source, ranges, ir.TensorRegionType())
    legalized = TensorCoreIntrinEmitter._legalize_to_buffer_region(region)
    assert legalized.same_as(region)
    assert legalized.source.same_as(source)
    assert [int(item.begin) for item in legalized.region] == [3, 5]
    full = TensorCoreIntrinEmitter._legalize_to_buffer_region(source)
    assert full.source.same_as(source)
    assert [int(item.extent) for item in full.region] == [32, 32]
    scalar = TensorCoreIntrinEmitter._legalize_to_buffer_region(ir.TensorLoad(source, [7, 11]))
    assert scalar.source.same_as(source)
    assert [int(item.begin) for item in scalar.region] == [7, 11]
    assert [int(item.extent) for item in scalar.region] == [1, 1]


@pytest.mark.parametrize("transpose_a,transpose_b", list(product([False, True], repeat=2)))
def test_mma_macros_construct_native_closed_ir(transpose_a, transpose_b, tmp_path):
    emitter = _emitter(a_transposed=transpose_a, b_transposed=transpose_b)
    a, b = _tensor("A", (32, 32), scope="shared"), _tensor("B", (32, 32), scope="shared")
    a_region = ir.TensorRegion(a, [ir.Range.from_min_extent(3, 16), ir.Range.from_min_extent(5, 16)], ir.TensorRegionType())
    b_region = ir.TensorRegion(b, [ir.Range.from_min_extent(7, 16), ir.Range.from_min_extent(11, 16)], ir.TensorRegionType())
    local_a = _tensor("a_local", (emitter.local_size_a,), scope="local")
    local_b = _tensor("b_local", (emitter.local_size_b,), scope="local")
    local_c = _tensor("c_local", (emitter.local_size_out,), "float32", scope="local")
    inputs = [a, b, local_a, local_b, local_c, emitter.thread_var]
    input_json = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.ldmatrix_a(local_a, a_region, 0)
        emitter.ldmatrix_b(local_b, b_region, 0)
        emitter.mma(local_a, local_b, local_c)

    closed, captures = close_lowered_function(body)
    assert [ir.save_json(value) for value in inputs] == input_json
    assert isinstance(closed, tirx.Function)
    assert tirx.analysis.check_well_formed(closed)
    assert not tirx.analysis.undefined_vars(closed)
    expected = [a, b, local_a, local_b, local_c, emitter.thread_var]
    assert all(any(value.same_as(capture) for capture in captures) for value in expected)
    calls = []
    tvm_ffi.structural_walk(closed.body, (ir.Call, calls.append), order="pre")
    names = [call.op.name for call in calls if isinstance(call.op, ir.Op)]
    assert names.count("tl.ptx_ldmatrix") == 2
    # TileLang's existing spelling is registered against the native PTX op.
    assert names.count("tirx.ptx_legacy.mma") == 2
    (tmp_path / "function.json").write_text(ir.save_json(closed))
    (tmp_path / "function.py").write_text(closed.script())


def test_mma_invalid_k_shape_keeps_padding_diagnostic():
    with pytest.raises(ValueError, match="must be divisible by 16.*Pad the K tile to 32"):
        _emitter(chunk=17)
