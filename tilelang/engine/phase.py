# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tvm import tir, IRModule
from tvm.target import Target
import tilelang as tl


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    mod = tir.transform.BindTarget(target)(mod)

    # Legalize the frontend IR to make it compatible with TVM
    mod = tl.transform.FrontendLegalize()(mod)
    # Simplify the IR expressions
    mod = tir.transform.Simplify()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tl.transform.LayoutInference()(mod)
    # Lower high-level tile operations to low-level operations
    mod = tl.transform.LowerTileOp()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tl.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tl.transform.LegalizeSafeMemoryAccess()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    mod = tir.transform.Simplify()(mod)

    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # which may be introduced by the LegalizeSafeMemoryAccess
    if target.arch == "sm_90":
        mod = tl.transform.MultiVersionBuffer()(mod)
        mod = tl.transform.WarpSpecialized()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tir.transform.LowerOpaqueBlock()(mod)
        mod = tl.transform.RewriteWgmmaSync()(mod)
        # mod = tl.transform.WarpSpecializedPipeline()(mod)
        mod = tl.transform.InjectFenceProxy()(mod)
    else:
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)

    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
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
    mod = tl.transform.ThreadPartialSync("shared.dyn")(mod)
    mod = tir.transform.InferFragment()(mod)
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    mod = tl.transform.ThreadSync("shared")(mod)
    mod = tl.transform.ThreadSync("shared.dyn")(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tl.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MergeSharedMemoryAllocations()(mod)

    mod = tl.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    return mod
