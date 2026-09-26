from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm import transform as tvm_transform
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)
from tilelang.metal.transform import MetalFragmentToSimdgroup

# Default loop unrolling for Metal. Apple's shader compiler does not unroll the
# short constant loops TileLang emits around simdgroup matrix operations on its
# own; a rolled loop keeps simdgroup fragments indexed by a runtime variable,
# which forces them out of registers, and the affected kernels measured several
# times slower. Mark short constant loops as unrolled so the Metal codegen emits
# an unroll pragma for them, the same division of labor as CUDA's pipeline and
# nvcc. An explicit ``tl.UnrollLoop`` pass configuration from the caller takes
# precedence.
METAL_UNROLL_LOOP = {"auto_max_step": 64, "auto_max_extent": 4, "explicit_unroll": False}


def _unroll_loops(mod: IRModule, pass_ctx: tvm_transform.PassContext) -> IRModule:
    if "tl.UnrollLoop" in pass_ctx.config:
        return tilelang.transform.UnrollLoop()(mod)
    config = dict(pass_ctx.config)
    config["tl.UnrollLoop"] = METAL_UNROLL_LOOP
    with tvm_transform.PassContext(
        opt_level=pass_ctx.opt_level,
        required_pass=list(pass_ctx.required_pass),
        disabled_pass=list(pass_ctx.disabled_pass),
        instruments=list(pass_ctx.instruments),
        config=config,
    ):
        return tilelang.transform.UnrollLoop()(mod)


def MetalPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_thread_binding=True, default_threads=128, unsupported_annotations=["cluster_dims"]
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

    # @Metal specific
    # On Metal, rewrite local.fragment GEMM accumulators to metal.simdgroup
    # before layout inference. simdgroup matrices are opaque and have no
    # explicit thread-level layout, so layout inference must not see them.
    mod = MetalFragmentToSimdgroup(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

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
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = _unroll_loops(mod, pass_ctx)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
