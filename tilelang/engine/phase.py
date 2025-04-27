# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tvm import tir, IRModule
from tvm.target import Target
import tilelang


def allow_tma_and_warp_specialized(target: Target) -> bool:
    if target.arch not in {"sm_90"}:
        return False
    cur_pass_ctx = tilelang.transform.get_pass_context()
    disable_tma_lower = cur_pass_ctx.config.get("tl.disable_tma_lower", False)
    disable_warp_specialized = cur_pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not (disable_tma_lower and disable_warp_specialized)


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    mod = tir.transform.BindTarget(target)(mod)

    # Legalize the frontend IR to make it compatible with TVM
    mod = tilelang.transform.FrontendLegalize()(mod)
    # Simplify the IR expressions
    mod = tir.transform.Simplify()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    mod = tir.transform.Simplify()(mod)
    # Try to vectorize loop with dynamic shape
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)

    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tir.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)
        mod = tilelang.transform.RewriteWgmmaSync()(mod)
        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tilelang.transform.ThreadPartialSync("shared.dyn")(mod)
    mod = tir.transform.InferFragment()(mod)
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    return mod
