from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_enable_race_check,
    should_force_let_inline,
)


def _should_enable_cpu_parallel(pass_ctx=None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_CPU_PARALLEL, False))


def CPUPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.cpu.transform.LowerCPUKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()
    cpu_parallel = _should_enable_cpu_parallel(pass_ctx)
    mod = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=False, unsupported_annotations=["cluster_dims"])(mod)

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
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    # Mark atomic kernels before LowerTileOp so grid parallelization stays serial.
    if cpu_parallel:
        mod = tilelang.cpu.transform.MarkCPUAtomics()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)
    # Lower remaining scalar atomics to serial RMW before vectorization.
    mod = tilelang.cpu.transform.LowerCPUAtomics()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # The CPU codegens have no native BF16. Host codegen legalizes BF16 storage
    # to uint16, which requires BF16 arithmetic to have been legalized first;
    # this is the point TVM's default pipeline runs it.
    mod = tirx.transform.BF16ComputeLegalize()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    if cpu_parallel:
        mod = tilelang.cpu.transform.MaterializeCPUParallelGrid()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    # CPU thread bindings are serial, so LowerThreadAllreduce is unnecessary.

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
