"""Explicit SIMD block iteration must not enter PTO lane vectorization."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang import tvm


def _kernel(mixed=False, automatic=False):
    dtype, vec, width = "float32", 64, 2048

    @T.prim_func
    def main(O: T.Tensor((2, width * 2), dtype)):
        with T.Kernel(1):
            temp = T.alloc_shared((2, width * 2), dtype)
            T.fill(temp, 123.0)
            with T.SimdVF():
                one = S.vdup(1.0, dtype)
                for h, block in T.Parallel(2, 32):
                    value = S.vld(temp[h, block * vec])
                    value = S.vadd(value, one)
                    S.vsts(temp[h, block * vec], value)
                    if mixed:
                        temp[h, width + block] = 1.0
                if automatic:
                    for j in T.Parallel(128):
                        temp[0, j] = T.cast(125.0, dtype)
            T.copy(temp, O)

    return main


@pytest.mark.parametrize("with_lanes", [False, True])
def test_explicit_simd_layout_preserves_loop_kind(with_lanes):
    from tilelang.ascend import transform as ascend
    from tilelang.ascend.target import normalize_pto_target

    mod = tvm.IRModule({"main": _kernel(mixed=True)})
    mod = tvm.tirx.transform.BindTarget(normalize_pto_target("pto"))(mod)
    lowered = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    lowered = tilelang.transform.AddWrapperForSingleBufStore()(lowered)
    if not with_lanes:

        def strip_lanes(node):
            if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
                return tvm.tirx.SBlock(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    node.body,
                    init=node.init,
                    alloc_buffers=node.alloc_buffers,
                    match_buffers=node.match_buffers,
                    annotations={key: value for key, value in node.annotations.items() if key != "tl.simdvf_lanes"},
                )

        lowered = tvm.IRModule(
            {
                "main": lowered["main"].with_body(
                    tvm.tirx.stmt_functor.ir_transform(lowered["main"].body, None, strip_lanes, ["tirx.SBlock"])
                )
            }
        )
    with normalize_pto_target("pto"):
        result = ascend.AscendLayoutInference()(lowered)
        ascend.VerifyParallelToPTO()(result)
    loops = []
    tvm.tirx.stmt_functor.post_order_visit(
        result["main"].body, lambda n: loops.append(n) if isinstance(n, tvm.tirx.For) and n.kind == tvm.tirx.ForKind.PARALLEL else None
    )
    assert len(loops) == 2


@pytest.mark.parametrize("automatic_sibling", [False, True])
def test_automatic_parallel_inside_explicit_simd_parallel_rejected(automatic_sibling):
    from tilelang.ascend import transform as ascend
    from tilelang.ascend.target import normalize_pto_target

    @T.prim_func
    def kernel():
        temp = T.alloc_shared((128,), "float32")
        with T.SimdVF():
            value = S.vdup(1.0, "float32")
            for block in T.Parallel(2):
                S.vsts(temp[block * 64], value)
                for j in T.Parallel(128):
                    temp[j] = T.float32(2.0)
            if automatic_sibling:
                for k in T.Parallel(128):
                    temp[k] = T.float32(3.0)

    with normalize_pto_target("pto") as target:
        mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": kernel}))
        with pytest.raises(
            tvm.error.InternalError, match="automatic T.Parallel nested inside an explicit SIMD T.Parallel.*not yet supported"
        ):
            ascend.VerifyParallelToPTO()(mod)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mixed", [False, True], ids=["explicit", "mixed"])
def test_explicit_simd_blocks_execute_once(mixed):
    output = tilelang.compile(_kernel(mixed), target="pto", out_idx=-1)()
    expected = torch.full((2, 4096), 123.0)
    expected[:, :2048] = 124.0
    if mixed:
        expected[:, 2048:2080] = 1.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_fp32_explicit_and_automatic_units_keep_both_codegen_paths():
    # Keep this a lowering check: this combined FP32/128-lane example exceeds
    # the current PTOAS/Bisheng VF stack budget during device compilation.
    from tilelang.ascend.target import normalize_pto_target
    from tilelang.engine.lower import lower

    with normalize_pto_target("pto") as target:
        source = lower(_kernel(automatic=True), target=target).kernel_source
    assert "pto.vmi." in source
    assert "pto.vlds" in source
    assert "pto.vsts" in source
