"""Native matrix-instruction contracts, independent of an installed ROCm SDK."""

from itertools import product

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang.language as T
from tilelang.rocm.intrinsics import MatrixCoreIntrinEmitter, WMMAIntrinEmitter
from tilelang.tileop._lowering import close_lowered_function


def _tensor(name, shape, dtype="float16", scope="local.fragment"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout="default"))


def _integer(value):
    return int(sym.Analyzer().simplify(value))


def _emitter(kind, **kwargs):
    options = dict(
        a_dtype=tvm_ffi.dtype("float16"),
        b_dtype=tvm_ffi.dtype("float16"),
        accum_dtype=tvm_ffi.dtype("float32"),
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,
        chunk=16,
        thread_var=ir.Var("tx", "int32"),
    )
    options.update(kwargs)
    if kind == "mfma":
        options.setdefault("target", Target({"kind": "hip", "mcpu": "gfx942"}))
        return MatrixCoreIntrinEmitter(**options)
    options.setdefault("target", Target({"kind": "hip", "mcpu": kind}))
    return WMMAIntrinEmitter(**options)


@pytest.mark.parametrize(
    "dtype,k_dim,suffix", [("float32", 4, "f32_16x16x4f32"), ("float16", 16, "f32_16x16x16f16"), ("int8", 32, "i32_16x16x32_i8")]
)
def test_mfma_native_dtype_dispatch(dtype, k_dim, suffix):
    out = "int32" if dtype == "int8" else "float32"
    emitter = _emitter("mfma", a_dtype=tvm_ffi.dtype(dtype), b_dtype=tvm_ffi.dtype(dtype), accum_dtype=tvm_ffi.dtype(out))
    assert emitter.k_dim == k_dim
    assert emitter.mfma_suffix == suffix


def test_mfma_gfx950_kpack_rule_applies_to_native_dtype():
    with pytest.warns(UserWarning, match="remapping"):
        emitter = _emitter("mfma", target=Target({"kind": "hip", "mcpu": "gfx950"}), k_pack=2, chunk=32)
    assert (emitter.k_dim, emitter.k_pack) == (32, 1)


@pytest.mark.parametrize("kind", ["mfma", "gfx1100", "gfx1200"])
def test_native_matrix_store_layout_matches_hardware_lane_slot(kind):
    emitter = _emitter(kind)
    tensor = _tensor("C", (16, 16), "float32")
    layout = emitter.make_mfma_store_layout(tensor) if kind == "mfma" else emitter.make_wmma_store_layout(tensor)
    seen = set()
    for row, column in product(range(16), repeat=2):
        if kind == "mfma":
            # MFMA issues B before A, so logical C uses the transposed D layout.
            expected = ((column // 4) * 16 + row, column % 4)
        elif kind == "gfx1100":
            # gfx11 interleaves adjacent M rows across the two half-waves.
            expected = ((row % 2) * 16 + column, row // 2)
        else:
            expected = ((row // 8) * 16 + column, row % 8)
        actual = (_integer(layout.map_forward_thread([row, column])[0]), _integer(layout.map_forward_index([row, column])[0]))
        assert actual == expected
        seen.add(actual)
    assert len(seen) == 256


@pytest.mark.parametrize("matrix,transpose", list(product(["A", "B"], [False, True])))
def test_wmma_gfx11_input_replication_preserves_both_halfwaves(matrix, transpose):
    emitter = _emitter("gfx1100", a_transposed=transpose, b_transposed=transpose)
    layout = emitter.make_wmma_load_layout(_tensor(matrix, (16, 16)), matrix)
    assert len(layout.get_forward_vars()) == 3
    for row, column in product(range(16), repeat=2):
        mn, k = (column, row) if (matrix == "A" and transpose) or (matrix == "B" and not transpose) else (row, column)
        # Fragment's explicit replication coordinate precedes logical coordinates.
        assert [_integer(layout.map_forward_thread([rep, row, column])[0]) for rep in (0, 1)] == [mn, mn + 16]
        assert all(_integer(layout.map_forward_index([rep, row, column])[0]) == k for rep in (0, 1))


@pytest.mark.parametrize("kind,transpose", list(product(["mfma", "gfx1100", "gfx1200"], [False, True])))
def test_rocm_matrix_macros_close_native_tensor_captures(kind, transpose, tmp_path):
    emitter = _emitter(kind, a_transposed=transpose, b_transposed=transpose)
    a, b = _tensor("A", (2, 32, 32), scope="shared"), _tensor("B", (2, 32, 32), scope="shared")
    ranges = [ir.Range.from_min_extent(1, 1), ir.Range.from_min_extent(3, 16), ir.Range.from_min_extent(5, 16)]
    ar, br = ir.TensorRegion(a, ranges, ir.TensorRegionType()), ir.TensorRegion(b, ranges, ir.TensorRegionType())
    la = _tensor("la", (emitter.local_size_a,), scope="local")
    lb = _tensor("lb", (emitter.local_size_b,), scope="local")
    lc = _tensor("lc", (emitter.local_size_out,), "float32", scope="local")
    operation = emitter.mfma if kind == "mfma" else emitter.wmma
    inputs = [a, b, la, lb, lc, emitter.thread_var]
    input_json = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.ldmatrix_a(la, ar, 0)
        emitter.ldmatrix_b(lb, br, 0)
        operation(la, lb, lc)

    closed, captures = close_lowered_function(body)
    assert [ir.save_json(value) for value in inputs] == input_json
    assert tirx.analysis.check_well_formed(closed)
    assert not tirx.analysis.undefined_vars(closed)
    assert all(any(item.same_as(capture) for capture in captures) for item in [a, b, la, lb, lc, emitter.thread_var])
    calls, loads = [], []
    tvm_ffi.structural_walk(closed.body, [(ir.Call, calls.append), (ir.TensorLoad, loads.append)])
    expected_op = "tl.tvm_mfma" if kind == "mfma" else "tl.tvm_rdna_wmma"
    matrix_calls = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == expected_op]
    assert len(matrix_calls) == 1
    shared_loads = [load for load in loads if load.source.same_as(a) or load.source.same_as(b)]
    assert len(shared_loads) == 2
    assert all(_integer(load.indices[0]) == 1 for load in shared_loads)
    (tmp_path / "function.py").write_text(closed.script())
    (tmp_path / "function.json").write_text(ir.save_json(closed))


@pytest.mark.parametrize("kind", ["mfma", "gfx1100"])
def test_rocm_region_legalization_uses_tensor_source_identity(kind):
    emitter = _emitter(kind)
    tensor = _tensor("A", (32, 32), scope="shared")
    load = ir.TensorLoad(tensor, [7, 11])
    region = emitter._legalize_to_buffer_region(load)
    assert region.source.same_as(tensor)
    assert [_integer(item.begin) for item in region.region] == [7, 11]
    assert [_integer(item.extent) for item in region.region] == [1, 1]
    assert emitter._legalize_to_buffer_region(region).same_as(region)
