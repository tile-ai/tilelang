import tvm_ffi


def StorageRewrite(enable_reuse=True, detect_inplace=True, preserve_shared_allocations=False, allow_local_inplace=False):
    return tvm_ffi.get_global_func("tl._tirx_passes.StorageRewrite")(
        enable_reuse, detect_inplace, preserve_shared_allocations, allow_local_inplace
    )


def OutlineDeviceRegions():
    return tvm_ffi.get_global_func("tl._tirx_passes.OutlineDeviceRegions")()


def LowerDeviceKernelLaunch():
    return tvm_ffi.get_global_func("tl._tirx_passes.LowerDeviceKernelLaunch")()


def SplitHostDevice():
    return tvm_ffi.get_global_func("tl._tirx_passes.SplitHostDevice")()


def VectorizeLoop(enable_vectorize=True):
    return tvm_ffi.get_global_func("tl._tirx_passes.VectorizeLoop")(enable_vectorize)


def HoistIf(allow_nontrivial_else=True):
    return tvm_ffi.get_global_func("tl._tirx_passes.HoistIf")(allow_nontrivial_else)


def FlattenTensor():
    return tvm_ffi.get_global_func("tl._tirx_passes.FlattenTensor")()


def LowerTensorLayouts():
    return tvm_ffi.get_global_func("tl._tirx_passes.LowerTensorLayouts")()


def StmtSimplify():
    return tvm_ffi.get_global_func("tl._tirx_passes.StmtSimplify")()


def RemoveNoOp():
    return tvm_ffi.get_global_func("tl._tirx_passes.RemoveNoOp")()


def MakePackedAPI():
    return tvm_ffi.get_global_func("tl._tirx_passes.MakePackedAPI")()


def BindTarget(target):
    return tvm_ffi.get_global_func("tl._tirx_passes.BindTarget")(target)


def AnnotateEntryFunc():
    return tvm_ffi.get_global_func("tl._tirx_passes.AnnotateEntryFunc")()


def LowerIntrin():
    return tvm_ffi.get_global_func("tl._tirx_passes.LowerIntrin")()


def LowerTVMBuiltin():
    return tvm_ffi.get_global_func("tl._tirx_passes.LowerTVMBuiltin")()


def ConvertSSA():
    return tvm_ffi.get_global_func("tl._tirx_passes.ConvertSSA")()


def NarrowDataType(target_bits=32):
    return tvm_ffi.get_global_func("tl._tirx_passes.NarrowDataType")(target_bits)


def ForceNarrowIndexToInt32():
    return tvm_ffi.get_global_func("tl._tirx_passes.ForceNarrowIndexToInt32")()
