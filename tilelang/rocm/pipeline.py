from __future__ import annotations

from tvm_tirx import IRModule, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def ROCMPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tilelang.transform.NormalizeAliasAnnotations()(mod)
    mod = tilelang.transform.LowerTensorSpec()(mod)
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True, lower_thread_binding=True, default_threads=128, unsupported_annotations=["cluster_dims"]
    )(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)
    # Warn on buffers that are read before anything writes them.
    # Runs after the reducer passes above, so legacy reducers have been
    # canonicalized, and before PipelinePlanning and LowerTileOp, while
    # loop bodies are still in source order and tile ops still declare
    # their access regions.
    mod = tilelang.transform.VerifyBufferInit()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.FinalizeAnnotations(target.kind.name)(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerTensorLayouts()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.StmtSimplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tirx.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.StmtSimplify()(mod)
    mod = tilelang.transform.RemoveNoOp()(mod)
    mod = tilelang.transform.HoistIf()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.InferWmmaFragmentMetadata()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.OutlineDeviceRegions()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # Device storage/synchronization must finish before resource metadata and
    # host launch operands are frozen. Packed argument lowering runs last.
    mod = tilelang.transform.Simplify()(mod)
    # Synchronization and fence passes still require the access extent/mask.
    # Erase that contract only when entering the final native address/launch ABI.
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    mod = tirx.transform.BF16ComputeLegalize()(mod)
    mod = tirx.transform.FP8ComputeLegalize()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tirx.transform.FP8StorageLegalize()(mod)
    mod = tirx.transform.BF16StorageLegalize()(mod)
    mod = tilelang.transform.HoistNonRestrictParams(finalize=True)(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    return mod
