"""Preserve pipeline hints, pending call effects, and barrier argument identity."""

import pytest
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.testing.ir import collect_nodes


def test_frontend_keeps_pending_barrier_calls_and_gemm_slot():
    @T.prim_func
    def kernel(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16")):
        shared_a = T.alloc_shared((16, 16), "float16")
        shared_b = T.alloc_shared((16, 16), "float16")
        accum = T.alloc_fragment((16, 16), "float32")
        barriers = T.alloc_barrier([1, 1])
        for index in T.serial(2):
            T.gemm(shared_a, shared_b, accum, mbar=barriers[index])
            T.mbarrier_wait_parity(barriers[index], index % 2)
        T.wait_wgmma(0)

    calls = collect_nodes(kernel, ir.Call)
    (gemm,) = [call for call in calls if call.op.name == "tl.tileop.gemm"]
    (wait,) = [call for call in calls if call.op.name == "tl.mbarrier_wait_parity"]
    (loop,) = collect_nodes(kernel, ir.For)
    assert isinstance(gemm.args[10], ir.TensorLoad)
    assert gemm.args[10].indices[0].same_as(loop.var)
    assert gemm.args[10].source.same_as(wait.args[0].source)
    assert wait.args[0].indices[0].same_as(loop.var)
    assert sum(call.op.name == "tl.wait_wgmma" for call in calls) == 1
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("loop_constructor", [T.serial, T.Parallel])
def test_pipeline_constants_remain_direct_loop_attributes(loop_constructor):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32")):
        for index in loop_constructor(
            4,
            annotations={
                "software_pipeline_stage": [T.int32(0)],
                "software_pipeline_order": [T.int32(0)],
                "tl_pipelined_num_stages": T.int32(1),
            },
        ):
            A[index] = 3

    (loop,) = collect_nodes(kernel, ir.For)
    assert list(loop.annotations["software_pipeline_stage"]) == [0]
    assert list(loop.annotations["software_pipeline_order"]) == [0]
    assert loop.annotations["tl_pipelined_num_stages"] == 1
    assert not collect_nodes(kernel, ir.RegionStmt)
    assert tirx.analysis.check_well_formed(kernel)
