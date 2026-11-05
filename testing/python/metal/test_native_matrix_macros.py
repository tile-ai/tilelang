"""Metal macro construction and layout/stride contracts without a Metal device."""

from itertools import product

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang.metal.language as T
from tilelang.metal.intrinsics import MPSIntrinEmitter
from tilelang.tileop._lowering import close_lowered_function


def _tensor(name, shape, dtype="float16", scope="local.fragment", strides=None):
    return ir.Var(name, T.Buffer(shape, dtype, scope=scope, strides=strides).as_type())


def _integer(value):
    return int(sym.Analyzer().simplify(value)) if isinstance(value, ir.Expr) else int(value)


def _emitter(cooperative=True, **kwargs):
    options = dict(
        warp_row_tiles=16 if cooperative else 8,
        warp_col_tiles=32 if cooperative else 8,
        thread_var=ir.Var("tx", "int32"),
        use_cooperative_tensor=cooperative,
    )
    options.update(kwargs)
    return MPSIntrinEmitter(**options)


def test_metal_region_keeps_pipeline_origin_and_affine_pitch():
    tensor = _tensor("A", (3, 32, 64), scope="shared", strides=(4096, 96, 1))
    region = ir.TensorRegion(tensor, [ir.Range.from_min_extent(i, n) for i, n in [(2, 1), (3, 16), (5, 32)]], ir.TensorRegionType())
    source, leading, row, column, stride = MPSIntrinEmitter._parse_buffer_nd(region)
    assert source.same_as(tensor)
    assert tuple(map(_integer, leading)) == (2,)
    assert tuple(map(_integer, (row, column, stride))) == (3, 5, 96)


def test_metal_noncontiguous_inner_dimension_keeps_diagnostic():
    tensor = _tensor("A", (32, 64), scope="shared", strides=(192, 2))
    with pytest.raises(ValueError, match="contiguous innermost dimension"):
        MPSIntrinEmitter._parse_buffer_nd(tensor)


def test_metal_cooperative_fragment_owner_register_bijection():
    emitter = _emitter()
    fragment = emitter.make_cooperative_tensor_store_layout(_tensor("C", (16, 32), "float32"))
    slots = set()
    for row, column in product(range(16), range(32)):
        group = column % 16 // 4
        expected_lane = group % 2 + (row % 4) * 2 + (group // 2) * 8 + (row % 8 // 4) * 16
        expected_slot = (column // 16) * 8 + (row // 8) * 4 + column % 4
        actual = (_integer(fragment.map_forward_thread([row, column])[0]), _integer(fragment.map_forward_index([row, column])[0]))
        assert actual == (expected_lane, expected_slot)
        slots.add(actual)
    assert slots == set(product(range(32), range(16)))


@pytest.mark.parametrize("cooperative,transpose", list(product([False, True], repeat=2)))
def test_metal_macro_constructs_native_load_mma_store(cooperative, transpose, tmp_path):
    emitter = _emitter(cooperative, a_transposed=transpose, b_transposed=transpose, block_row_warps=2, block_col_warps=2)
    a, b = _tensor("A", (2, 64, 64), scope="shared"), _tensor("B", (2, 64, 64), scope="shared")
    c = _tensor("C", (64, 64), "float32", scope="shared")
    ranges = [ir.Range.from_min_extent(1, 1), ir.Range.from_min_extent(3, 32), ir.Range.from_min_extent(5, 32)]
    ar, br = ir.TensorRegion(a, ranges, ir.TensorRegionType()), ir.TensorRegion(b, ranges, ir.TensorRegionType())
    la, lb = _tensor("la", (16,), scope="local"), _tensor("lb", (16,), scope="local")
    lc = _tensor("lc", (16,), "float32", scope="local")
    inputs = [a, b, c, la, lb, lc, emitter.thread_var]
    input_json = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.ldmatrix_a(la, ar, 0)
        emitter.ldmatrix_b(lb, br, 0)
        emitter.mma(la, lb, lc)
        emitter.simdgroup_copy(lc, c)

    closed, captures = close_lowered_function(body)
    assert [ir.save_json(value) for value in inputs] == input_json
    assert tirx.analysis.check_well_formed(closed)
    assert not tirx.analysis.undefined_vars(closed)
    assert all(any(item.same_as(capture) for capture in captures) for item in [a, b, c, la, lb, lc, emitter.thread_var])
    calls = []
    tvm_ffi.structural_walk(closed.body, (ir.Call, calls.append))
    names = [call.op.name for call in calls if isinstance(call.op, ir.Op)]
    prefix = "tl.cooperative_tensor_" if cooperative else "tirx.metal.simdgroup_"
    assert sum(name.endswith("load") and prefix in name for name in names) == 2
    assert sum(name.endswith("store") and prefix in name for name in names) == 1
    assert sum(name.endswith("multiply_accumulate") and prefix in name for name in names) == 1
    (tmp_path / "function.py").write_text(closed.script())
    (tmp_path / "function.json").write_text(ir.save_json(closed))
