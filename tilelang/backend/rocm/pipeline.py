from __future__ import annotations

from tvm import IRModule, tir
from tvm.target import Target

import tilelang
from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.engine.pass_pipeline import (
    LayoutVisual,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def lower_amd(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    pass_ctx = tilelang.transform.get_pass_context()

    mod = tilelang.transform.LowerSharedTmem()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MarkCudaSyncCalls(False)(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=should_enable_aggressive_merge(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


rocm_pipeline = Pipeline("hip", lower_amd)

register_pipeline(rocm_pipeline)
