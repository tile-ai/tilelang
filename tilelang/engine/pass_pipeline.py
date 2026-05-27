from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.transform import PassContext


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tirx.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)


def should_disable_shared_memory_reuse(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE, False))


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """Run backend-independent validation before lowering."""

    if not should_enable_prelower_semantic_check():
        return

    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    tilelang.analysis.NestedLoopChecker()(mod)
    tilelang.analysis.FragmentLoopChecker()(mod)


def CommonPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Common lowering pipeline."""

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

    # @CUDA-specific
    # Tile-level warp specialization: runs before layout inference so that
    # producer/consumer split happens at the high-level tile-op IR.
    # The pass classifies copy ops as TMA/cp.async/sync inline (no prior
    # InstructionAnnotation pass needed). Shared buffers are multi-versioned
    # internally only for functions where the WS transformation actually
    # applies.
    # if allow_warp_specialized(target=target):
    #     mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)

    # @CUDA / Blackwell specific
    # Lower 2SM TCGEN5MMA and related on Blackwell target (must run before
    # LayoutInference so that the use_2cta annotation is visible to infer_layout)
    # mod = tilelang.transform.LowerBlackwell2SM()(mod)

    # Normalize if-without-else wrappers before pipeline planning. This keeps
    # pipeline body extraction focused on canonical SeqStmt bodies.
    mod = tilelang.transform.IfStmtBinding()(mod)

    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # @Metal specific
    # On Metal, rewrite local.fragment GEMM accumulators to metal.simdgroup
    # before layout inference. simdgroup matrices are opaque and have no
    # explicit thread-level layout, so layout inference must not see them.
    # mod = tilelang.transform.metal.MetalFragmentToSimdgroup(mod)

    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)

    # @CUDA specific
    # Lower l2 persistent map
    # mod = tilelang.cuda.transform.LowerL2Persistent()(mod)
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

    # @CUDA-specific
    # Lower the shared.tmem into specific initialization slot
    # mod = tilelang.transform.LowerSharedTmem()(mod)

    # HasTMA
    # has_tma = module_has_tma(mod)

    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # @CUDA-specific
    # mod = tilelang.transform.LowerSharedBarrier()(mod)

    # @CUDA-specific
    # if has_tma:
    #    mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)

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

    # @CUDA-specific
    # mod = tilelang.transform.LowerLDGSTG()(mod)
    # mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # @CUDA-specific
    # Mark the function contains pdl_sync or pdl_trigger
    # mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    # @CUDA-specific
    # InjectFenceProxy is a no-op on targets that lack the TMA / async-proxy
    # programming model; the pass itself checks the PrimFunc's target.
    # mod = tilelang.transform.InjectFenceProxy()(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)

    # @CUDA-specific
    # Inject conservative tcgen05 fences on Blackwell (SM100+).
    # Must run after ThreadSync so that tvm_storage_sync calls are present.
    # The pass handles shared syncs and simple linear wait/use, use/arrive
    # handoffs, and is a no-op on non-SM100 targets or functions without TMEM.
    # mod = tilelang.transform.InjectTcgen05Fence()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)

    # @CUDA-specific
    # NOTE: LowerPTXAsyncCopy is applied earlier (before PipelinePlanning).
    # if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
    #     mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)

    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # @CUDA-specific
    # Transform threadblock to persistent threadblock
    # mod = tilelang.cuda.transform.PersistThreadblock()(mod)

    return mod
