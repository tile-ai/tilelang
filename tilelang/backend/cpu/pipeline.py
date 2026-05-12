from __future__ import annotations

from tvm import tir, IRModule
from tvm.target import Target

from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.engine.phase import PreLowerSemanticCheck


def cpu_lower_and_legalize(mod: IRModule, target: Target) -> IRModule:
    """CPU-specific lower and legalize pipeline.

    A simplified version of the GPU pipeline that skips TMA, warp
    specialization, Blackwell-2SM, pipeline planning, and other
    GPU-only passes.
    """
    import tilelang

    mod = tir.transform.BindTarget(target)(mod)

    if tilelang.engine.phase.should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if tilelang.engine.phase.should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    tilelang.engine.phase.LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def cpu_optimize_for_target(mod: IRModule, target: Target) -> IRModule:
    """CPU-specific optimize for target pipeline.

    Skips GPU-only passes (TMA, thread sync, shared memory merging,
    Hopper intrinsics, etc.).
    """
    import tilelang

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
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=tilelang.engine.phase.allow_vectorize(pass_ctx=pass_ctx))(mod)
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
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MarkCudaSyncCalls(False)(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)
    mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.InjectTcgen05Fence()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


# Register CPU pipelines for both "c" and "llvm" target kinds
for _kind in ("c", "llvm"):
    register_pipeline(
        Pipeline(_kind)
        .set_pre_lower_semantic_check(PreLowerSemanticCheck)
        .set_lower_and_legalize(cpu_lower_and_legalize)
        .set_optimize_for_target(cpu_optimize_for_target)
    )
