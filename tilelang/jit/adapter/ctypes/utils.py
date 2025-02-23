# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
from typing import Union, Optional
from tilelang import tvm as tvm
from tvm import IRModule, tir
from tvm.target import Target
import tilelang.transform
from tilelang.engine.lower import (
    is_device_call,
    determine_target,
    canon_target_host,
)


def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) >= 1  # may have statement before kernel
    return source.index(matched[0])


def is_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda"


def is_hip_target(target: Target) -> bool:
    return target.kind.name == "hip"


def get_annotated_device_mod(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
) -> "IRModule":

    mod = func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    mod = tir.transform.BindTarget(target)(mod)

    mod = tilelang.transform.FrontendLegalize()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tir.transform.Simplify()(mod)

    if target.arch == "sm_90":
        mod = tilelang.transform.WarpSpecializedPipeline()(mod)
    else:
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)

    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tir.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tir.transform.ThreadSync("shared")(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tir.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tir.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tir.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    device_mod = tir.transform.Filter(is_device_call)(mod)

    return device_mod
