from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc, SBlock
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)
from tilelang.contrib.nvcc import (
    get_target_compute_version,
    have_mbarrier,
    have_pdl,
    have_tma,
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


def _module_has_shared_barrier(mod: IRModule) -> bool:
    """Whether any function allocates a shared.barrier / shared.cluster_barrier
    buffer (i.e. uses ``T.alloc_barrier``).
    """
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() in ("shared.barrier", "shared.cluster_barrier"):
                    found = True

    for _, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            post_order_visit(func.body, visit)
    return found


def CUDAPassPipelineBodyPrologue(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
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
    # The pass classifies copy ops as TMA/cp.async/sync inline. Shared buffers
    # are multi-versioned internally only for functions where the WS
    # transformation actually applies.
    if allow_warp_specialized(target=target):
        mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)

    # @CUDA / Blackwell specific
    # Lower 2SM TCGEN5MMA and related on Blackwell target (must run before
    # LayoutInference so that the use_2cta annotation is visible to infer_layout)
    mod = tilelang.cuda.transform.LowerBlackwell2SM()(mod)

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

    # @CUDA specific
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
    return mod


def CUDAPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()

    mod = CUDAPassPipelineBodyPrologue(mod, target)

    # @CUDA-specific
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.cuda.transform.LowerSharedTmem()(mod)

    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # @CUDA-specific
    # LowerSharedBarrier emits hardware mbarrier code (the cutlass Barrier type,
    # tl::tl_shuffle_elect, tl::fence_barrier_init) which only exists on sm_90+.
    # Reject T.alloc_barrier() up front on pre-Hopper targets instead of letting
    # nvcc fail later with cryptic "identifier 'Barrier' is undefined" errors.
    if not have_mbarrier(target) and _module_has_shared_barrier(mod):
        compute_version = get_target_compute_version(target)
        raise ValueError(
            f"T.alloc_barrier() requires sm_90 (Hopper) or later, but the current "
            f"target is sm_{compute_version.replace('.', '')} (compute capability "
            f"{compute_version}). Hardware mbarrier operations (Barrier type, "
            f"tl_shuffle_elect, fence_barrier_init) are not available on this "
            f"architecture. Use __syncthreads() or named barriers for pre-Hopper "
            f"targets."
        )
    mod = tilelang.cuda.transform.LowerSharedBarrier()(mod)

    # @CUDA-specific
    has_tma = module_has_tma(mod)
    if has_tma:
        mod = tilelang.cuda.transform.FuseMBarrierArriveExpectTx()(mod)

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
    mod = tilelang.cuda.transform.LowerLDGSTG()(mod)
    mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # @CUDA-specific
    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.cuda.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    # @CUDA-specific
    # InjectFenceProxy is a no-op on targets that lack the TMA / async-proxy
    # programming model; the pass itself checks the PrimFunc's target.
    mod = tilelang.cuda.transform.InjectFenceProxy()(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)

    # @CUDA-specific
    # Inject conservative tcgen05 fences on Blackwell (SM100+).
    # Must run after ThreadSync so that tvm_storage_sync calls are present.
    # The pass handles shared syncs and simple linear wait/use, use/arrive
    # handoffs, and is a no-op on non-SM100 targets or functions without TMEM.
    mod = tilelang.cuda.transform.InjectTcgen05Fence()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)

    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(mod)

    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # @CUDA-specific
    # Transform threadblock to persistent threadblock
    mod = tilelang.cuda.transform.PersistThreadblock()(mod)

    return mod


cuda_pipeline = PassPipeline("cuda", CUDAPassPipelineBody)

register_pipeline(cuda_pipeline)
