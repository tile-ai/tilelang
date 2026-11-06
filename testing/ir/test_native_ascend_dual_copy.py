"""Native dual-copy rewriting preserves subcore scope, regions, and hardware paths."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.ascend.op  # noqa: F401 -- use the real Ascend GEMM callbacks.
from tilelang.ascend import language as T
from tilelang.layout import Layout
from testing.ascend._ir import substitute
from testing.ascend._block import attribute_fields, has_attribute

from test_native_ascend_ops import function, strided_tensor, subregion
from test_native_cuda_copy import calls
from test_native_reducer_contracts import annotations, attribute, decoded, integer, objects, run_pass, tensor, tileop


def _rewrite(value, automatic=True):
    before = ir.save_json(value)
    with tvm_tirx.transform.PassContext(config={"tl.enable_auto_schedule": automatic}):
        result = run_pass("RewriteDualCopy", value)
        assert tirx.analysis.check_well_formed(result)
        assert not tirx.analysis.undefined_vars(result)
        ir.assert_structural_equal(result, run_pass("RewriteDualCopy", result))
    assert ir.save_json(value) == before
    return result


def _launch(body, tag="cthread", extent=2, var=None):
    var = var if var is not None else ir.Var("program_index", "int32")
    extent = integer(extent) if isinstance(extent, int) else extent
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), extent], [var], {}, body)


def _block(name, body):
    return attribute(None, "tl.sblock", {"name": name, "no_realize": True, "annotations": {"keep": 7}}, body)


def _subcores(value):
    found = []
    for node in objects(value, ir.RegionStmt):
        if node.op.name == "tirx.launch_thread" and node.args[0].value == "cthread":
            found.append((node.body_params[0], node.args[1]))
        elif node.op.name == "tl.launch_kernel":
            for index, tag in enumerate(node.attrs["grid_tags"]):
                if tag == "cthread":
                    found.append((node.body_params[index], node.args[1].fields[index]))
    return found


def _attributes(value, key):
    return [attribute_fields(node)["value"] for node in objects(value, ir.RegionStmt) if has_attribute(node, key)]


def _copy(src_scope, dst_scope, src_shape, dst_shape, *, split=1, double=False):
    src, dst = tensor("src", src_shape, "float32", src_scope), tensor("dst", dst_shape, "float32", dst_scope)
    metadata = {"dual_dst_ctl": split, "unit_flag_ctrl": 3, "l2_cache_ctrl": 2}
    if double:
        metadata["double"] = 1
    operation = tileop(
        "ascend_copy", subregion(src, [0] * len(src_shape), src_shape), subregion(dst, [0] * len(dst_shape), dst_shape), metadata=metadata
    )
    return src, dst, operation


def _materialize(value):
    before = ir.save_json(value)
    with value.attrs["target"]:
        result = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(True, False, None, ["cluster_dims"], ["cthread"])(
            ir.IRModule({"main": value})
        )["main"]
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    return result


def _evaluate(expression, values):
    analyzer = tvm_tirx.sym.Analyzer()
    return int(analyzer.simplify(substitute(expression, values)))


@pytest.mark.parametrize(
    "src_scope,dst_scope,src_shape,dst_shape,double,split,axis",
    [
        ("global", "shared.dyn", (128,), (64,), False, 1, 0),
        ("shared.dyn", "global", (64,), (128,), True, 1, 0),
        ("shared.dyn", "shared.l1", (64,), (128,), True, 1, 0),
        ("global", "shared.dyn", (64, 64), (32, 64), False, 1, 0),
        ("shared.dyn", "global", (64, 32), (64, 64), True, 2, 1),
    ],
)
@pytest.mark.parametrize("manual", [False, True])
def test_software_copy_splits_full_region_using_the_actual_subcore(src_scope, dst_scope, src_shape, dst_shape, double, split, axis, manual):
    src, dst, operation = _copy(src_scope, dst_scope, src_shape, dst_shape, split=split, double=double)
    scope = _launch([operation])
    sid = scope.body_params[0]
    if manual:
        scope = _block("VECTOR", [scope])
    result = _rewrite(function([src, dst], [scope]), not manual)
    (copy,) = calls(result, "tl.tileop.ascend_copy")
    operands, metadata = decoded(copy)
    full, half = (operands[1], operands[0]) if double else (operands[0], operands[1])
    extent = half.region[axis].extent
    ir.assert_structural_equal(full.region[axis].begin, sid * extent)
    ir.assert_structural_equal(full.region[axis].extent, extent)
    assert all(int(r.begin) == 0 for i, r in enumerate(full.region) if i != axis)
    assert not {"double", "dual_dst_ctl", "unit_flag_ctrl"} & set(metadata)
    assert int(metadata["l2_cache_ctrl"]) == 2
    assert operands[0].source.same_as(src) and operands[1].source.same_as(dst)


def test_automatic_copy_synthesizes_one_native_subcore_scope():
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    body = _launch([operation, operation], "blockIdx.x", 1)
    result = _rewrite(function([src, dst], [body]))
    (scope,) = _subcores(result)
    assert int(scope[1]) == 2
    assert len(calls(result, "tl.tileop.ascend_copy")) == 2


@pytest.mark.parametrize("split,src_shape,dst_shape", [(1, (64, 64), (32, 64)), (2, (64, 64), (64, 32))])
@pytest.mark.parametrize("automatic", [False, True])
def test_hardware_copy_retains_dual_destination_unit_flags_and_no_subcore(split, src_shape, dst_shape, automatic):
    src, dst, operation = _copy("shared.l0c", "shared.dyn", src_shape, dst_shape, split=split)
    value = function([src, dst], [operation])
    result = _rewrite(value, automatic)
    ir.assert_structural_equal(result, value)
    assert not _subcores(result)


@pytest.mark.parametrize(
    "src_scope,dst_scope,src_shape,dst_shape,split,message",
    [
        ("global", "shared.l1", (64, 64), (32, 64), 1, "supports only L0C->UB"),
        ("shared.l0c", "shared.dyn", (127, 64), (63, 64), 1, "exact 2:1 extent ratio"),
        ("shared.l0c", "shared.dyn", (64, 48), (64, 24), 2, "multiple of 32"),
        ("shared.l0c", "shared.dyn", (128,), (64,), 1, "at least two-dimensional"),
        ("global", "shared.dyn", (127,), (64,), 1, "exact 2:1 extent ratio"),
    ],
)
def test_invalid_copy_regions_keep_the_original_diagnostics(src_scope, dst_scope, src_shape, dst_shape, split, message):
    src, dst, operation = _copy(src_scope, dst_scope, src_shape, dst_shape, split=split)
    value = function([src, dst], [_launch([operation])])
    before = ir.save_json(value)
    with pytest.raises(ValueError, match=message):
        _rewrite(value)
    assert ir.save_json(value) == before


@pytest.mark.parametrize("mixed", [False, True])
def test_manual_copy_requires_an_explicit_vector_scope(mixed):
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    body = _launch([operation]) if mixed else operation
    with pytest.raises(ValueError, match=r"explicit T\.Vector"):
        _rewrite(function([src, dst], [body]), False)


@pytest.mark.parametrize("extent", [1, 4, "symbolic"])
def test_software_copy_requires_exactly_two_subcores(extent):
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    n = ir.Var("n", "int32")
    actual = n if extent == "symbolic" else extent
    value = function([src, dst, n], [_launch([operation], extent=actual)])
    with pytest.raises(ValueError, match="cthread extent of 2"):
        _rewrite(value)


@pytest.mark.parametrize("manual", [False, True])
def test_unscoped_copy_cannot_invent_a_kernel_boundary(manual):
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    body = _block("VECTOR", [operation]) if manual else operation
    with pytest.raises(ValueError, match="outside a cthread"):
        _rewrite(function([src, dst], [body]), not manual)


def test_manual_mixed_scope_remains_invalid_inside_a_nested_vector():
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    nested = _launch([_block("VECTOR", [_launch([operation])])])
    with pytest.raises(ValueError, match=r"explicit T\.Vector"):
        _rewrite(function([src, dst], [nested]), False)


def test_nested_subcore_and_sibling_kernel_scopes_restore_their_sid():
    src, dst, operation = _copy("global", "shared.dyn", (128,), (64,))
    inner = _launch([operation])
    first = _launch([inner, operation], "blockIdx.x", 1)
    second = _launch([operation], "blockIdx.x", 1)
    result = _rewrite(function([src, dst], [first, second]))
    scopes = _subcores(result)
    assert len(scopes) == 3
    outer_sid, inner_sid, second_sid = [item[0] for item in scopes]
    assert len({var for var, _ in scopes}) == 3
    expected = [inner_sid, outer_sid, second_sid]
    for operation, sid in zip(calls(result, "tl.tileop.ascend_copy"), expected):
        operands, _ = decoded(operation)
        ir.assert_structural_equal(operands[0].region[0].begin, sid * 64)


@pytest.mark.parametrize("path", ["gm-to-ub", "ub-to-gm", "ub-to-l1"])
def test_dynamic_subregions_preserve_origins_strides_and_other_metadata(path):
    source_full = path == "gm-to-ub"
    src_scope = "global" if source_full else "shared.dyn"
    dst_scope = {"gm-to-ub": "shared.dyn", "ub-to-gm": "global", "ub-to-l1": "shared.l1"}[path]
    src = strided_tensor("src", [8, 128], [160, 1], 13, "float32", src_scope)
    dst = strided_tensor("dst", [8, 128], [192, 1], 17, "float32", dst_scope)
    n = ir.Var("n", "int32")
    src_width, dst_width = (2 * n, n) if source_full else (n, 2 * n)
    metadata = {"dual_dst_ctl": 2, "unit_flag_ctrl": 3, "extra": {"source": src, "n": n}, "l2_cache_ctrl": 2}
    if not source_full:
        metadata["double"] = 1
    operation = tileop("ascend_copy", subregion(src, [1, 3], [4, src_width]), subregion(dst, [2, 5], [4, dst_width]), metadata=metadata)
    scope = _launch([operation])
    sid = scope.body_params[0]
    result = _rewrite(function([src, dst, n], [scope]))
    (copy,) = calls(result, "tl.tileop.ascend_copy")
    operands, remaining = decoded(copy)
    full = operands[0 if source_full else 1]
    for count in (0, 3, 17):
        for subcore in (0, 1):
            assert _evaluate(full.region[1].begin, {n: count, sid: subcore}) == (3 if source_full else 5) + subcore * count
            assert _evaluate(full.region[1].extent, {n: count, sid: subcore}) == count
    assert full.source.same_as(src if source_full else dst)
    assert operands[0].source.same_as(src) and operands[1].source.same_as(dst)
    ir.assert_structural_equal(remaining["extra"]["source"], src)
    ir.assert_structural_equal(remaining["extra"]["n"], n)
    assert int(remaining["l2_cache_ctrl"]) == 2


def _dual_dsl(mode, annotated=False):
    kernel = T.MixedKernel if mode == "mixed" else T.Kernel
    prelude = "// dual-copy annotation owner" if annotated else None
    layout = Layout([32], lambda i: i)

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with kernel(1, prelude=prelude):
            temporary = T.alloc_shared((32,), "float32")
            if annotated:
                T.annotate_layout({temporary: layout})
            if mode == "manual":
                with T.Vector():
                    T.dual_copy(A, temporary)
                    T.dual_copy(temporary, B)
            else:
                T.dual_copy(A, temporary)
                T.dual_copy(temporary, B)

    return main.with_attr("target", tvm_tirx.target.Target("ascend")).with_attr("global_symbol", "main")


@pytest.mark.parametrize("mode", ["automatic", "mixed", "manual"])
@pytest.mark.parametrize("materialize_before", [False, True])
def test_real_dsl_dual_copy_survives_native_launch_materialization_and_lowering(mode, materialize_before):
    value = _dual_dsl(mode)
    if materialize_before:
        value = _materialize(value)
    result = _rewrite(value, automatic=mode != "manual")
    (scope,) = _subcores(result)
    assert int(scope[1]) == 2
    if not materialize_before:
        result = _materialize(result)
    for operation in calls(result, "tl.tileop.ascend_copy"):
        operands, metadata = decoded(operation)
        assert [int(region.region[0].extent) for region in operands] == [32, 32]
        assert not {"double", "dual_dst_ctl", "unit_flag_ctrl"} & set(metadata)
    for name in ("AscendLayoutInference", "NormalizeAscendFractalStorage", "AscendInsertOOBPadding", "AscendLowerTileOp"):
        result = run_pass(name, result)
        assert tirx.analysis.check_well_formed(result)
    assert len(calls(result, "tl.ascend_copy_gm_to_ubuf")) == 1
    assert len(calls(result, "tl.ascend_copy_ubuf_to_gm")) == 1
    assert not [call for call in calls(result) if call.op.name.startswith("tl.tileop.")]
    # Core dispatch metadata belongs to the later core/source pipeline. It
    # must survive this pass rather than disappear to make emission succeed.
    if mode == "mixed":
        assert [int(value["vector_count"]) for value in _attributes(result, "tl.kernel")] == [2]
    if mode == "manual":
        assert [int(value["annotations"]["vector_count"]) for value in _attributes(result, "tl.sblock") if value["name"] == "VECTOR"] == [2]


@pytest.mark.parametrize("mode", ["automatic", "mixed", "manual"])
@pytest.mark.parametrize("annotated", [False, True])
def test_rewrite_commutes_with_native_kernel_materialization(mode, annotated):
    value = _dual_dsl(mode, annotated)
    after = _materialize(_rewrite(value, mode != "manual"))
    before = _rewrite(_materialize(value), mode != "manual")
    ir.assert_structural_equal(after, before)
    if annotated:
        for candidate in (after, before):
            (subcore,) = _subcores(candidate)
            assert int(subcore[1]) == 2
            hints = [metadata["pragma_import_c"] for metadata in _attributes(candidate, "tl.kernel") if "pragma_import_c" in metadata]
            assert list(map(str, hints)) == ["// dual-copy annotation owner"]
            layouts = [metadata["layout_map"] for metadata in annotations(candidate) if "layout_map" in metadata]
            assert len(layouts) == 1 and len(layouts[0]) == 1
            (temporary,) = [bind.var for bind in objects(candidate, ir.Bind) if bind.var.name == "temporary"]
            assert next(iter(layouts[0])).same_as(temporary)
        for name in ("AscendLayoutInference", "NormalizeAscendFractalStorage", "AscendInsertOOBPadding", "AscendLowerTileOp"):
            after, before = run_pass(name, after), run_pass(name, before)
            ir.assert_structural_equal(after, before)


def _original_gemm():
    relative = "examples/ascend/example_gemm.py"
    source = Path(__file__).resolve().parents[2] / relative
    repository = Path(os.environ.get("TILELANG_OPS_EXAMPLE_REPOSITORY", source.parents[2]))
    baseline = subprocess.check_output(["git", "show", "baceed6676dfe6c37878b845993d4ee03f409b58:" + relative], cwd=repository)
    assert source.read_bytes() == baseline
    assert hashlib.sha256(baseline).hexdigest() == "288ee516f7ffaafc66bc879075400e1972c163b6bc7c193b3a8dcaec40aa56dd"
    specification = importlib.util.spec_from_file_location("native_dual_original_gemm", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "dtype,out_dtype,mixed,hf32,accumulate",
    [
        ("bfloat16", "bfloat16", False, None, False),
        ("bfloat16", "float32", True, None, False),
        ("float32", "float32", False, "nearest_even", False),
        ("float8_e4m3fn", "float32", True, None, False),
        ("bfloat16", "float32", True, None, True),
    ],
)
def test_original_gemm_source_passes_real_native_dual_layout_storage_and_lowering(dtype, out_dtype, mixed, hf32, accumulate, tmp_path):
    example = _original_gemm()
    value = example.gemm(512, 512, 512, dtype=dtype, out_dtype=out_dtype, MIXED=mixed, hf32=hf32, acc=accumulate)
    value = value.with_attr("target", tvm_tirx.target.Target("ascend")).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(value)
    before = ir.save_json(value)
    result = _rewrite(_materialize(value))
    assert len(_subcores(result)) == int(mixed)
    for name in ("AscendLayoutInference", "NormalizeAscendFractalStorage", "AscendInsertOOBPadding", "AscendLowerTileOp"):
        result = run_pass(name, result)
        assert tirx.analysis.check_well_formed(result)
        assert not tirx.analysis.undefined_vars(result)
    assert calls(result, "tl.ascend_gemm_l1")
    assert len(calls(result, "tl.ascend_copy_gm_to_cbuf")) == 2
    assert not [call for call in calls(result) if call.op.name.startswith("tl.tileop.")]
    if mixed:
        assert calls(result, "tl.ascend_copy_matrix_cc_to_ub")
        assert calls(result, "tl.ascend_copy_ubuf_to_gm")
    assert ir.save_json(value) == before
    (tmp_path / "lowered.json").write_text(ir.save_json(result))
