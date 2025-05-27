# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma
from typing import Optional


def allow_warp_specialized(pass_ctx: Optional[PassContext] = None,
                           target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: Optional[PassContext] = None,
                                   target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target) or not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    return is_cuda_target(target) and have_tma(target)


def allow_vectorize(pass_ctx: Optional[PassContext] = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


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
    pass_ctx = tilelang.transform.get_pass_context()
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        mod = tilelang.transform.InjectTmaBarrier()(mod)
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        # warp_specialized pass will pack the if stmt into the block
        # so we need to lower the opaque block first
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

        if allow_fence_proxy(target=target):
            # in hopper device, wgmma is an async proxy
            # so we need to inject a fence proxy before it
            mod = tilelang.transform.InjectFenceProxy()(mod)

    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

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

    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)

    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        mod = tir.transform.MergeSharedMemoryAllocations()(mod)
    else:
        mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.EliminateStorageSyncForMBarrier()(mod)
    # Inject PTX async copy must behind the thread sync pass
    # as ptx async copy won't be recognized as a valid buffer load
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)

    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    return mod
