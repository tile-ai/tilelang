from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.contrib.nvcc import have_pdl, have_tma
from tilelang.engine.pass_pipeline import (
    LayoutVisual,
    allow_vectorize,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
    should_disable_shared_memory_reuse,
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
    """Check if any function in the module was lowered with TMA operations.

    This reads the ``tl.has_tma`` attribute set by ``LowerTileOp`` during
    ``LowerAndLegalize``, which is the source of truth for whether TMA
    copies were actually generated.
    """
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in mod.functions.items())


def CUDAPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Tile-level warp specialization: runs before layout inference so that
    # producer/consumer split happens at the high-level tile-op IR.
    # The pass classifies copy ops as TMA/cp.async/sync inline (no prior
    # InstructionAnnotation pass needed). Shared buffers are multi-versioned
    # internally only for functions where the WS transformation actually
    # applies.
    if allow_warp_specialized(target=target):
        mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)
    # Lower 2SM TCGEN5MMA and related on Blackwell target (must run before
    # LayoutInference so that the use_2cta annotation is visible to infer_layout)
    mod = tilelang.transform.LowerBlackwell2SM()(mod)
    # Normalize if-without-else wrappers before pipeline planning. This keeps
    # pipeline body extraction focused on canonical SeqStmt bodies.
    mod = tilelang.transform.IfStmtBinding()(mod)
    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Lower l2 persistent map
    mod = tilelang.cuda.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    has_tma = module_has_tma(mod)
    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    if has_tma:
        mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
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

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)
    # InjectFenceProxy is a no-op on targets that lack the TMA / async-proxy
    # programming model; the pass itself checks the PrimFunc's target.
    mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    # Inject conservative tcgen05 fences on Blackwell (SM100+).
    # Must run after ThreadSync so that tvm_storage_sync calls are present.
    # The pass handles shared syncs and simple linear wait/use, use/arrive
    # handoffs, and is a no-op on non-SM100 targets or functions without TMEM.
    mod = tilelang.transform.InjectTcgen05Fence()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # NOTE: LowerPTXAsyncCopy is applied earlier (before PipelinePlanning).
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Transform threadblock to persistent threadblock
    mod = tilelang.cuda.transform.PersistThreadblock()(mod)
    return mod


cuda_pipeline = Pipeline("cuda", CUDAPassPipelineBody)

register_pipeline(cuda_pipeline)
