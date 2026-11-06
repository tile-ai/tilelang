"""The Ascend layout policy follows native VF scopes without changing defaults."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T

from test_native_ascend_ops import function
from test_native_ascend_sf_lowering import gemm_case
from test_native_reducer_contracts import allocate, annotation, integer, loop, objects, region, run_pass, tensor, tileop, update


def attributes(value, name):
    result = []
    for scope in objects(value, ir.RegionStmt):
        vf = {"tl.simdvf_scope": "SIMD_VF", "tl.simtvf_scope": "SIMT_VF"}.get(name)
        if vf and scope.op.name == "tl.ascend_block" and scope.attrs["name"] == vf:
            result.append((scope, True))
        if scope.op.name == "tl.attr":
            data = tvm_ffi.get_global_func("tl.DecodeAnnotations")(scope.args[0], scope.attrs["annotation_schema"])
            if data["key"] == name:
                result.append((scope, data["value"]))
    return result


def infer(value):
    before = ir.save_json(value)
    result = run_pass("AscendLayoutInference", value)
    assert before == ir.save_json(value)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    return result


def vf_program():
    @T.prim_func
    def main(source: T.Tensor((2, 64), "float32")):
        temporary = T.alloc_shared((2, 64), "float32")
        for i, j in T.Parallel(2, 64):
            temporary[i, j] = source[i, j]
        with T.SimtVF(32):
            for i, j in T.Parallel(2, 64):
                temporary[i, j] = source[i, j]
            with T.SimdVF():
                for i, j in T.Parallel(2, 64):
                    temporary[i, j] = source[i, j]
            for i, j in T.Parallel(2, 64):
                temporary[i, j] = source[i, j]
        for i, j in T.Parallel(2, 64):
            temporary[i, j] = source[i, j]

    return main.with_attr("target", tvm_tirx.target.Target("ascend"))


def test_simd_body_stays_opaque_and_simt_threads_restore_after_nested_exit():
    value = vf_program()
    result = infer(value)
    (before,), (after,) = attributes(value, "tl.simdvf_scope"), attributes(result, "tl.simdvf_scope")
    ir.assert_structural_equal(before[0], after[0])
    assert len(objects(after[0], ir.For)) == 2
    assert not attributes(after[0], "tl.loop.annotations")
    layouts = [data["parallel_loop_layout"] for _, data in attributes(result, "tl.loop.annotations")]
    assert [int(layout.get_thread_size()) for layout in layouts] == [1, 32, 32, 1]
    assert len(objects(result, ir.For)) == 6


def test_default_layout_pass_keeps_its_previous_behavior_for_the_same_program():
    value = vf_program()
    result = run_pass("LayoutInference", value)
    (scope,) = attributes(result, "tl.simdvf_scope")
    assert len(objects(scope[0], ir.For)) == 1
    assert len(attributes(scope[0], "tl.loop.annotations")) == 1
    assert len(objects(result, ir.For)) == 5


@pytest.mark.parametrize("marker", [False, True])
def test_only_simt_reducer_nests_skip_fusion(marker):
    source = tensor("source", [2, 16], "float32", "global")
    acc = tensor("acc", [2], "float32", "local.reducer")
    result = tensor("result", [2], "float32", "local.fragment")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    body = [
        allocate(acc),
        allocate(result),
        annotation({"reducer_info_v2": {acc: {"op": "sum"}}}),
        tileop("reducer_init", region(acc)),
        loop(i, 2, [loop(j, 16, [update(acc, [i], ir.TensorLoad(source, [i, j]))])]),
        tileop("finalize_reducer_v2", region(acc), region(result)),
    ]
    tx = ir.Var("tx", "int32")
    scope = ir.RegionStmt("tl.ascend_block", [], [], {"name": "SIMT_VF"}, body) if marker else ir.SeqStmt(body)
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(32)], [tx], {}, [scope])
    value = function([source], [launch])
    for name in ("CanonicalizeLegacyReducer", "VerifyReducerEpoch"):
        value = run_pass(name, value)
    result = infer(value)
    loops = objects(result, ir.For)
    assert len(loops) == (2 if marker else 1)
    assert [len(data["parallel_loop_layout"].get_input_shape()) for _, data in attributes(result, "tl.loop.annotations")] == [
        2 if marker else 1
    ]


def test_plain_native_dma_uses_the_default_layout_engine():
    gm, ub = tensor("gm", [8, 16], "float16", "global"), tensor("ub", [8, 16], "float16", "shared.dyn")
    value = function([gm], [allocate(ub), tileop("ascend_copy", region(gm), region(ub))])
    ir.assert_structural_equal(infer(value), run_pass("LayoutInference", value))


@pytest.mark.parametrize("case", ["valid", "compact"])
@pytest.mark.parametrize("symbolic", [False, True])
def test_sf_gemm_uses_real_layout_storage_and_lowering_passes(case, symbolic):
    value, _, _, _ = gemm_case(case, symbolic=symbolic)
    result = infer(value)
    assert "tl.l0_sf_bindings" in ir.save_json(result)
    for name in ("NormalizeAscendFractalStorage", "AscendInsertOOBPadding", "AscendLowerTileOp"):
        result = run_pass(name, result)
        assert tirx.analysis.check_well_formed(result)
    assert "tl.l0_sf_bindings" not in ir.save_json(result)


def test_invalid_sf_gemm_fails_at_layout_boundary_before_mutation():
    value, _, _, _ = gemm_case("wrong-data")
    before = ir.save_json(value)
    with pytest.raises(Exception, match="is bound to"):
        run_pass("AscendLayoutInference", value)
    assert ir.save_json(value) == before
