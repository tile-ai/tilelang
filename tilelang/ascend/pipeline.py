from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_race_check,
    should_force_let_inline,
)

from . import transform as ascend_transform


def allow_autoschedule(pass_ctx=None) -> bool:
    """Whether the Ascend auto-scheduler should run for this pass context.

    Ascend-owned: TL_ENABLE_AUTO_SCHEDULE only exists when the Ascend backend is
    compiled in, so this does not belong on the backend-neutral
    pipeline_utils surface. AutoSchedule schedules Ascend per-core tasks, so
    disabling it leaves the kernel unscheduled rather than handing it to another
    scheduler.
    """
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE, True)


def AscendPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    # Materialize the target-neutral kernel-launch nest emitted by T.Kernel.
    # Ascend's NPU launch is a real 1-D blockIdx.x core grid, so the grid loop
    # becomes a thread_extent AttrStmt; there is no threadIdx at kernel scope,
    # so the `tx/ty/tz = tl.launch_thread_idx(...)` placeholders are dropped.
    # Thread domains only exist inside T.SimtVF, which emits its own thread
    # scopes below the launch nest. `cthread` is Ascend's sub-block-id launch
    # dimension, emitted by T.MixedKernel and read back by the Ascend codegen.
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline(pass_ctx=pass_ctx):
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check(pass_ctx=pass_ctx):
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    # Expand AIV-side dual copies before layout, OOB, and scheduling passes.
    # Reuse an enclosing cthread sid from an explicit T.Vector block; an
    # auto-scheduled mixed T.Kernel gets one synthesized before its core
    # sections are materialized. L0C->UB remains a hardware dual-destination
    # copy on AIC.
    mod = ascend_transform.RewriteDualCopy()(mod)
    # MODE_MERGING preserves inactive destination lanes, so it is a
    # read-modify-write operation rather than a pure value expression. Expose
    # the destination as an rw access before scheduling and dependency analysis.
    mod = ascend_transform.LegalizeSimdMerging()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)
    mod = tilelang.transform.VerifyBufferInit()(mod)

    # Materialize only user-requested explicit unrolls outside VF blocks so
    # LayoutInference and AutoSchedule can consume the expanded operations.
    mod = ascend_transform.UnrollLoopSkipVF()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = ascend_transform.AscendLayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    mod = ascend_transform.NormalizeAscendFractalStorage()(mod)

    # Rewrite UB->L1 copies into NZ buffers as scatter+post_copy, using the NZ
    # layouts just inferred. Must run before AutoSchedule so it can schedule the
    # emitted ops.
    mod = ascend_transform.InsertNd2Nz()(mod)

    # Run VFChecker after InsertNd2Nz: the UB->L1 copies it would otherwise
    # reject (e.g. casting copies) are by now rewritten into a SIMD_VF scatter.
    tilelang.ascend.analysis.VFChecker()(mod)

    # Clamp DMA copy OOB tails and emit GM->L1 padding as semantic T.fill ops so
    # AutoSchedule sees each fill's exact L1 write region. LowerTileOp converts
    # them to ascend_fill_l1 after scheduling, which codegen emits as
    # asc_fill_l1. Must run before AutoSchedule.
    mod = ascend_transform.AscendInsertOOBPadding()(mod)

    if allow_autoschedule(pass_ctx=pass_ctx):
        mod = ascend_transform.NormalizeControlFlowForSchedule()(mod)
        mod = ascend_transform.NormalizeConflictHints()(mod)
        mod = ascend_transform.MaterializeScheduleUnits()(mod)
        mod = ascend_transform.AnnotateMultiBufferEligible()(mod)
        mod = ascend_transform.EstimateLatency()(mod)
        mod = ascend_transform.AutoSchedule()(mod)
        mod = ascend_transform.AssignCore()(mod)
        mod = ascend_transform.PrepareMultiBuffer()(mod)
        mod = ascend_transform.ResolveCore()(mod)
        mod = ascend_transform.InsertSync()(mod)
        mod = ascend_transform.MaterializeMultiBuffer()(mod)
        mod = ascend_transform.LowerScheduledTIR()(mod)
        mod = ascend_transform.RestoreWhileLoops()(mod)

    mod = tilelang.transform.Simplify()(mod)

    mod = ascend_transform.NormalizeBufferVersion()(mod)
    mod = ascend_transform.AscendSimdVFLowerParallel()(mod)
    mod = ascend_transform.AscendLowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = ascend_transform.RewriteAscendBufferVersionLayout()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    # Retype packed 4-bit float (float4_e2m1fn) storage into the 1-byte
    # packed-pair form (float4_e2m1fnx2) and halve fp4 offsets/indices, so
    # codegen only ever sees a valid 1-byte fp4 element. Must run after the
    # access_ptr/copy offsets are materialized and before MergeUBAllocations.
    mod = ascend_transform.RewriteFp4ToFp4x2()(mod)

    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = ascend_transform.AscendRemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse().passes[0](mod)
    mod = tirx.transform.Simplify()(mod)
    mod = ascend_transform.AscendRemoveNoOp()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    if allow_global_thread_synchronization(pass_ctx=pass_ctx):
        mod = ascend_transform.AscendThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = ascend_transform.MarkScalarDcacheBypass()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = ascend_transform.MergeUBAllocations(
        align_bytes=32,
        disable_reuse=disable_reuse,
    )(mod)

    # Split each hard_event between its 8-slot flag namespace and the shared
    # get_buf/rls_buf mutex pool (knapsack) to minimize wasted flag_ids; a no-op
    # when no hard_event overflows 8 slots.
    mod = ascend_transform.RewriteFlagToBuf()(mod)

    mod = ascend_transform.AscendThreadSync("shared")(mod)
    mod = ascend_transform.AscendThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)
