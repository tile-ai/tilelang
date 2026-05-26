from __future__ import annotations

from tilelang import tvm
import tilelang as tl
from tvm.target import Target

from tilelang.backend.cuda.pipeline import allow_warp_specialized
from tilelang.engine.pass_pipeline import (
    LayoutVisual,
    should_enable_race_check,
    should_force_let_inline,
)


def lower_cuda_until_tileop(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Run the CUDA lowering prefix used by transform unit tests."""
    mod = tvm.tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        mod = tl.transform.LetInline()(mod)
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tl.transform.VerifyParallelLoop()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LayoutReducer()(mod)
    if allow_warp_specialized(target=target):
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tl.transform.LowerBlackwell2SM()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tl.cuda.transform.LowerL2Persistent()(mod)
    mod = tl.transform.DecoupleTypeCast()(mod)
    mod = tl.transform.LegalizeVectorizedLoop()(mod)
    mod = tl.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tl.transform.LowerAccessPtr()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.HoistNonRestrictParams()(mod)
    return mod
