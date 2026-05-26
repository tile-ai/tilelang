from __future__ import annotations

from tvm import tirx, s_tir, IRModule
from tvm.target import Target

import tilelang
from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.contrib.nvcc import have_pdl, have_tma
from tilelang.engine.pass_pipeline import (
    LayoutVisual,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)
from tilelang.transform import PassContext


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # Avoid importing jit.adapter.utils at module import time.
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def module_has_tma(mod: IRModule) -> bool:
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in mod.functions.items())


def lower_cuda(mod: IRModule, target: Target) -> IRModule:
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
    if allow_warp_specialized(target=target):
        mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tilelang.transform.LowerBlackwell2SM()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.cuda.transform.LowerL2Persistent()(mod)
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    pass_ctx = tilelang.transform.get_pass_context()
    has_tma = module_has_tma(mod)

    mod = tilelang.transform.LowerSharedTmem()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    if has_tma:
        mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)
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
    mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx)
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        enable_aggressive_merge = False
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)

    mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.InjectTcgen05Fence()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    mod = tilelang.cuda.transform.PersistThreadblock()(mod)
    return mod


cuda_pipeline = Pipeline("cuda", lower_cuda)

register_pipeline(cuda_pipeline)
