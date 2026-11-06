"""SF slot handles disappear only after their real native consumers lower."""

import os
from pathlib import Path

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.ascend.op  # noqa: F401 -- register the production GEMM lowerers.
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_l0c_layout, make_ascend_major_k_layout, make_ascend_major_mn_layout, make_ascend_sf_layout

from test_native_ascend_ops import address_offset, constant, emit_source, function, strided_tensor, subregion
from test_native_cuda_copy import calls
from test_native_reducer_contracts import allocate, annotation, attribute, integer, objects, region, run_pass, tensor, tileop


def lower(value, name, *, infer=True):
    original = ir.save_json(value)
    inferred = run_pass("LayoutInference", value) if infer else value
    result = run_pass("AscendLowerTileOp", inferred)
    assert ir.save_json(value) == original
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    assert not any(call.op.name.startswith("tl.tileop.") for call in calls(result))
    assert not sf_variables(result)
    assert "tl.l0_sf_bindings" not in ir.save_json(result)
    if directory := os.environ.get("TILELANG_OPS_ARTIFACT_DIR"):
        (Path(directory) / (name + ".json")).write_text(ir.save_json(result))
    return result


def sf_variables(value):
    return [
        var
        for var in objects(value, ir.Var)
        if isinstance(var.ty, (tirx.TensorType, ir.PtrType)) and var.ty.scope.name in ("shared.l0a.sf", "shared.l0b.sf")
    ]


def contiguous_strides(shape):
    result, stride = [], 1
    for extent in reversed(shape):
        result.insert(0, stride)
        stride *= extent
    return result


@pytest.mark.parametrize(
    "leading,index,scope,transposed,dtype,pack",
    [
        ((), (), "shared.l0a", False, "uint16", 1),
        ((3,), (1,), "shared.l0b", False, "uint16", 1),
        ((2, 3), (1, 2), "shared.l0a", False, "uint8", 2),
        ((2, 3), (1, 2), "shared.l0b", True, "uint16", 1),
        ((), (), "shared.l0a", True, "uint8", 2),
    ],
)
def test_sf_copy_lowers_to_bound_data_address_before_retirement(leading, index, scope, transposed, dtype, pack, tmp_path):
    shape = [*leading, *((128, 64) if transposed else (64, 128))]
    data = strided_tensor("data", shape, contiguous_strides(shape), 13, "float8_e4m3fn", scope)
    sf = tensor("sf", [*leading, 64, 2 * pack], dtype, scope + ".sf")
    source = strided_tensor("source", [64, 4 * pack], [4 * pack, 1], 7, dtype, "shared.l1")
    layouts = {
        source: make_ascend_sf_layout(source),
        data: (make_ascend_major_mn_layout if transposed else make_ascend_major_k_layout)(data),
    }
    value = function(
        [],
        [
            allocate(source),
            allocate(data),
            allocate(sf),
            annotation({"layout_map": layouts, "tl.l0_sf_bindings": {sf: data}}),
            tileop(
                "ascend_copy", subregion(source, [16, pack], [32, pack]), subregion(sf, [*index, 0, 0], [*([1] * len(leading)), 32, pack])
            ),
        ],
    )
    result = lower(value, "sf-copy-" + scope + str(leading) + dtype + str(transposed))
    (copy,) = calls(result, "tl.ascend_load_ca_sf" if scope.endswith("a") else "tl.ascend_load_cb_sf")
    tile = 0
    for size, position in zip(leading, index):
        tile = tile * size + position
    assert address_offset(copy.args[0]) == 13 + tile * 64 * 128
    assert address_offset(copy.args[1]) == 7
    assert [constant(arg) for arg in copy.args[2:]] == [1, 1, 2, 1, 4, 1]
    # LayoutInference/LowerTileOp consume layout_map into native TensorTypes.
    # Observe the complete physical shape, origin, and every basis stride.
    allocated = {bind.var.name: bind.var for bind in objects(result, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)}
    assert set(allocated) == {"data", "source"}
    for name, expected_shape, origin in (
        ("data", [*leading, 4, 4, 16, 32], 13),
        ("source", [4, 4, 16, pack], 7),
    ):
        physical = allocated[name]
        assert [constant(dim) for dim in physical.ty.shape] == expected_shape
        assert physical.ty.layout is not None
        zeros = [0] * len(expected_shape)
        assert constant(physical.ty.layout.apply(*zeros, shape=physical.ty.shape)["m"]) == origin
        for axis, stride in enumerate(contiguous_strides(expected_shape)):
            if expected_shape[axis] == 1:
                continue
            basis = [int(index == axis) for index in range(len(expected_shape))]
            assert constant(physical.ty.layout.apply(*basis, shape=physical.ty.shape)["m"]) == origin + stride
    code = emit_source(result, tmp_path)
    assert "asc_copy_l12l0" in code and "_mx(" in code


def gemm_case(case="valid", *, symbolic=False):
    a_shape = [2, 3, 32, 128]
    a = strided_tensor("a", a_shape, contiguous_strides(a_shape), 11, "float8_e4m3fn", "shared.l0a")
    a2 = tensor("a2", a_shape, "float8_e4m3fn", "shared.l0a")
    b = tensor("b", [32, 128], "float8_e4m3fn", "shared.l0b")
    c = tensor("c", [32, 32], "float32", "shared.l0c")
    sf = tensor("sf", [2, 3, 32, 2], "uint16", "shared.l0b.sf" if case == "scope" else "shared.l0a.sf")
    sb = tensor("sb", [32, 2], "uint16", "shared.l0b.sf")
    allocations = [a, a2, b, c, sf, sb]
    bindings = {sf: a2 if case == "wrong-data" else a, sb: b}
    if case == "missing":
        del bindings[sf]
    if case == "duplicate":
        duplicate = tensor("sf_duplicate", [2, 3, 32, 2], "uint16", "shared.l0a.sf")
        allocations.append(duplicate)
        bindings[duplicate] = a
    m = ir.Var("m", "int32") if symbolic else 17
    n = ir.Var("n", "int32") if symbolic else 19
    k = 64 if case == "compact" else 128
    ar = subregion(a, [1, 2, 0, 0], [1, 1, m, k])
    sr = subregion(sf, [0 if case == "leading" else 1, 2, 1 if case == "origin" else 0, 0], [1, 1, m, 1 if case in ("k", "compact") else 2])
    br, cr, sbr = subregion(b, [0, 0], [n, k]), subregion(c, [0, 0], [m, n]), subregion(sb, [0, 0], [n, 1 if case == "compact" else 2])
    call = T.gemm_blockscaled(ar, br, cr, sr, sbr, transpose_B=True, clear_accum=True)
    value = function(
        [m, n] if symbolic else [],
        [
            *(allocate(var) for var in allocations),
            annotation(
                {
                    "tl.l0_sf_bindings": bindings,
                    "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
                }
            ),
            ir.Evaluate(call),
        ],
    )
    return value, m, n, k


@pytest.mark.parametrize("case", ["valid", "compact"])
@pytest.mark.parametrize("symbolic", [False, True])
def test_real_blockscaled_gemm_consumes_sf_bindings_and_retains_data_storage(case, symbolic, tmp_path):
    value, m, n, k = gemm_case(case, symbolic=symbolic)
    result = lower(value, "sf-gemm-" + case + str(symbolic))
    (mad,) = calls(result, "tl.ascend_mad_mx")
    assert address_offset(mad.args[1]) == 11 + 5 * 32 * 128
    assert constant(mad.args[4]) == k
    ir.assert_structural_equal(mad.args[3], m if symbolic else integer(m))
    ir.assert_structural_equal(mad.args[5], n if symbolic else integer(n))
    allocated = {bind.var.name: bind.var for bind in objects(result, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)}
    assert constant(allocated["a"].ty.layout.apply(*([0] * len(allocated["a"].ty.shape)), shape=allocated["a"].ty.shape)["m"]) == 11
    assert len(allocated["a"].ty.shape) == 6
    assert "asc_mmad_mx" in emit_source(result, tmp_path)


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("wrong-data", "is bound to"),
        ("missing", "no allocation binding"),
        ("scope", "matching L0A/L0B"),
        ("leading", "same leading index"),
        ("origin", "zero-origin"),
        ("k", "compact data K extent"),
        ("duplicate", "multiple SF handles"),
    ],
)
def test_invalid_sf_binding_and_slice_fail_without_mutating_input(case, diagnostic):
    value, _, _, _ = gemm_case(case)
    original = ir.save_json(value)
    with pytest.raises(Exception, match=diagnostic):
        run_pass("AscendLowerTileOp", value)
    assert ir.save_json(value) == original
    assert sf_variables(value)


@pytest.mark.parametrize("carrier", ["annotation", "attribute", "kernel"])
def test_retirement_preserves_other_metadata_and_lexical_scope(carrier):
    data = tensor("data", [16, 64], "float8_e4m3fn", "shared.l0a")
    sf = tensor("sf", [16, 1], "uint16", "shared.l0a.sf")
    marker = ir.Var("marker", "int32")
    local = tensor("local", [4], "float32", "local")
    body = [allocate(local), ir.TensorStore(local, [0], ir.const(2, "float32"))]
    metadata = {"tl.l0_sf_bindings": {sf: data}, "test_marker": marker}
    if carrier == "annotation":
        suffix = [annotation(metadata), *body]
    elif carrier == "attribute":
        suffix = [annotation({"test_marker": marker}), attribute(None, "tl.l0_sf_bindings", {sf: data}, body)]
    else:
        suffix = [attribute(None, "tl.kernel", metadata, body)]
    result = lower(function([marker], [allocate(data), allocate(sf), *suffix]), "sf-metadata-" + carrier, infer=False)
    assert "test_marker" in ir.save_json(result)
    assert any(var.same_as(marker) for var in objects(result, ir.Var))
    if carrier == "attribute":
        assert any(stmt.op.name == "tl.lexical_scope" for stmt in objects(result, ir.RegionStmt))
    # Generic user hints retire only at finalization. Their containing native
    # definitions and side effects must survive; internal contracts still
    # require their owning pass (covered by test_native_unknown_annotations).
    before = ir.save_json(result)
    finalized = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(ir.IRModule({"main": result}))["main"]
    assert ir.save_json(result) == before
    assert "test_marker" not in ir.save_json(finalized)
    assert tirx.analysis.check_well_formed(finalized)
    assert not tirx.analysis.undefined_vars(finalized)
    old_stores, new_stores = objects(result, ir.TensorStore), objects(finalized, ir.TensorStore)
    assert len(old_stores) == len(new_stores) == 1
    ir.assert_structural_equal(old_stores[0], new_stores[0])
    assert old_stores[0].dest.same_as(new_stores[0].dest)
    if carrier in ("attribute", "kernel"):
        assert any(stmt.op.name == "tl.lexical_scope" for stmt in objects(finalized, ir.RegionStmt))


@pytest.mark.parametrize("use", ["load", "pointer", "foreign_metadata"])
def test_live_sf_use_prevents_retirement(use):
    data = tensor("data", [16, 64], "float8_e4m3fn", "shared.l0a")
    sf = tensor("sf", [16, 1], "uint16", "shared.l0a.sf")
    consumer = {
        "load": ir.Evaluate(ir.TensorLoad(sf, [0, 0])),
        "pointer": ir.Evaluate(sf.data),
        "foreign_metadata": annotation({"foreign_metadata": sf}),
    }[use]
    value = function([], [allocate(data), allocate(sf), annotation({"tl.l0_sf_bindings": {sf: data}}), consumer])
    original = ir.save_json(value)
    with pytest.raises(ValueError, match="unlowered consumer"):
        run_pass("AscendLowerTileOp", value)
    assert ir.save_json(value) == original


def test_unbound_unused_sf_handle_is_not_silently_erased():
    sf = tensor("sf", [16, 1], "uint16", "shared.l0a.sf")
    value = function([], [allocate(sf)])
    original = ir.save_json(value)
    with pytest.raises(ValueError, match="no allocation binding"):
        run_pass("AscendLowerTileOp", value)
    assert ir.save_json(value) == original


def test_ordinary_dma_uses_the_common_native_lowering_unchanged():
    gm, ub = tensor("gm", [4, 8], "float16", "global"), tensor("ub", [4, 8], "float16", "shared.dyn")
    value = run_pass("LayoutInference", function([gm], [allocate(ub), tileop("ascend_copy", region(gm), region(ub))]))
    ir.assert_structural_equal(run_pass("LowerTileOp", value), run_pass("AscendLowerTileOp", value))


def test_native_simtvf_scope_selects_simt_copy_and_restores_dma_after_exit():
    import tvm_tirx

    @T.prim_func
    def kernel(source: T.Tensor((64,), "float32")):
        ub = T.alloc_shared((64,), "float32")
        with T.SimtVF(32):
            T.copy(source, ub)
        T.copy(source, ub)

    value = kernel.with_attr("target", tvm_tirx.target.Target("ascend")).with_attr("global_symbol", "main")
    result = lower(value, "ascend-native-simtvf-context")
    scopes = []
    for stmt in objects(result, ir.RegionStmt):
        if stmt.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(stmt.args[0], stmt.attrs["annotation_schema"])
            if fields["key"] == "tl.simtvf_scope":
                scopes.append(stmt)
    (scope,) = scopes
    assert not calls(scope, "tl.ascend_copy_gm_to_ubuf")
    assert objects(scope, ir.TensorLoad) and objects(scope, ir.TensorStore)
    assert len(calls(result, "tl.ascend_copy_gm_to_ubuf")) == 1
