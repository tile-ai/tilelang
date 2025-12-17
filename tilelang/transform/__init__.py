"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .simplify import Simplify, simplify_prim_func, LetInline  # noqa: F401
from .pass_config import PassConfigKey  # noqa: F401
from tilelang import tvm as tvm  # noqa: F401
from tvm.ir.transform import PassContext  # noqa: F401
from .add_bufstore_wrapper import AddWrapperForSingleBufStore  # noqa: F401


def get_pass_context():
    """Get the current pass context"""
    return PassContext.current()


def ClusterPlanning():
    """ClusterPlanning

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ClusterPlanning()  # type: ignore


def PipelinePlanning():
    """infer the fragment/shared memory layout

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PipelinePlanning()  # type: ignore


def LayoutInference():
    """LayoutInference

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """LowerTileOp

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTileOp()  # type: ignore


def InjectSoftwarePipeline():
    """InjectSoftwarePipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def FrontendLegalize():
    """FrontendLegalize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FrontendLegalize()  # type: ignore


def LegalizeNegativeIndex():
    """Legalize negative indices in buffer loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeNegativeIndex()  # type: ignore


def InjectAssumes():
    """Inject Assumes for natural shape boundary conditions. And convert Assumes in Evaluate(Call(...)) form
    (tvm builtin assume call) to AttrNode form.

    Returns:
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectAssumes()


def LowerHopperIntrin():
    """LowerHopperIntrin

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerHopperIntrin() if hasattr(_ffi_api, "LowerHopperIntrin") else lambda f: f  # type: ignore


def WarpSpecializedPipeline():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.WarpSpecializedPipeline()  # type: ignore


def RewriteWgmmaSync():
    """RewriteWgmmaSync

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteWgmmaSync()  # type: ignore


def ThreadSync(storage_scope: str):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def ThreadPartialSync(storage_scope: str):
    """Insert partial sync.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadPartialSync(storage_scope)  # type: ignore


def IfStmtBinding():
    """IfStmtBinding

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.IfStmtBinding()  # type: ignore


def MergeIfStmt():
    """MergeIfStmt

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeIfStmt()  # type: ignore


def MultiVersionBuffer():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MultiVersionBuffer()  # type: ignore


def WarpSpecialized():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.WarpSpecialized()  # type: ignore


def AnnotateWarpGroupRegAlloc():
    """Inject set_max_nreg calls into warp-specialized functions.

    This pass analyzes the function to collect register hints from set_max_nreg
    and no_set_max_nreg calls, then injects appropriate set_max_nreg calls into
    producer and consumer branches of warp-specialized code.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateWarpGroupRegAlloc()  # type: ignore


def InjectTmaBarrier():
    """InjectTmaBarrier

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectTmaBarrier()  # type: ignore


def InjectFenceProxy():
    """InjectFenceProxy

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectFenceProxy()  # type: ignore


def LegalizeVectorizedLoop():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeVectorizedLoop()  # type: ignore


def LegalizeSafeMemoryAccess():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeSafeMemoryAccess()  # type: ignore


def MakePackedAPI():
    """MakePackedAPI

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def AnnotateDeviceRegions():
    """AnnotateDeviceRegions

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def SplitHostDevice():
    """Split host/device functions even for empty kernels.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def AnnotateReadOnlyParams():
    """Annotate read-only handle parameters for PrimFuncs.

    Adds attribute `tl.readonly_param_indices` listing param indices that are
    never written, enabling CUDA codegen to emit `const` qualifiers to unlock
    read-only cache loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateReadOnlyParams()  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """VectorizeLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def InjectPTXAsyncCopy():
    """Rewrite global to shared memory copy on CUDA with asynchronous copy.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPTXAsyncCopy()  # type: ignore


def LowerDeviceStorageAccessInfo():
    """Lower attached storage access information on device.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    Run this pass after all storage access analysis finish.
    """
    return _ffi_api.LowerDeviceStorageAccessInfo()  # type: ignore


def ConfigIndexBitwidth():
    """Config index bitwidth.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.ConfigIndexBitwidth()  # type: ignore


def FlattenBuffer():
    """FlattenBuffer

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def EliminateStorageSyncForMBarrier():
    """EliminateStorageSyncForMBarrier"""
    return _ffi_api.EliminateStorageSyncForMBarrier()  # type: ignore


def MergeSharedMemoryAllocations(enable_aggressive_merge: bool = False, align_bytes: int = 16):
    """MergeSharedMemoryAllocations

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations(enable_aggressive_merge, align_bytes)  # type: ignore


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


def AlignDynamicSharedMemoryAllocations(align_bytes: int = 16):
    """AlignDynamicSharedMemoryAllocations

    Parameters
    ----------
    align_bytes: int
        The alignment bytes.

    Returns
    -------
    """
    return _ffi_api.AlignDynamicSharedMemoryAllocations(align_bytes)  # type: ignore


def LowerSharedBarrier():
    """LowerSharedBarrier"""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def PlanAndUpdateBufferAllocationLocation():
    """Plan and update buffer allocation locations within PrimFuncs.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()  # type: ignore


def HoistNonRestrictParams():
    return _ffi_api.HoistNonRestrictParams()  # type: ignore


def StorageRewrite():
    """StorageRewrite

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def LowerOpaqueBlock():
    """LowerOpaqueBlock"""
    return _ffi_api.LowerOpaqueBlock()  # type: ignore


def LowerThreadAllreduce():
    """LowerThreadAllreduce"""
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerIntrin():
    """LowerIntrin"""
    return _ffi_api.LowerIntrin()  # type: ignore


def LowerDeviceKernelLaunch():
    """
    Create and return a transform pass that lowers device kernel launch constructs to target-specific IR.

    This pass transforms high-level device kernel launch and related intrinsics into lower-level
    IR suitable for backend code generation and device-side lowering.

    Returns:
        tvm.transform.Pass: The transform pass that performs device kernel launch lowering.
    """
    return _ffi_api.LowerDeviceKernelLaunch()  # type: ignore


def LowerSharedTmem():
    """LowerSharedTmem"""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LayoutReducer():
    """
    Return a TVM transform pass that performs layout reduction/normalization.

    This wrapper delegates to the underlying FFI implementation and returns a pass object suitable for use in a PassContext or pass pipeline. The pass is intended to simplify or reduce tensor/layout-related representations during relay/tile transformations.

    Returns:
        The transform pass object produced by the FFI backend.
    """
    return _ffi_api.LayoutReducer()  # type: ignore
