# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .simplify import Simplify, simplify_prim_func  # noqa: F401
from .pass_config import PassConfigKey  # noqa: F401
from tilelang import tvm as tvm  # noqa: F401
from tvm.ir.transform import PassContext  # noqa: F401


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


def LowerHopperIntrin():
    """LowerHopperIntrin

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerHopperIntrin() \
        if hasattr(_ffi_api, "LowerHopperIntrin") else lambda f: f # type: ignore


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


def LoopVectorizeDynamic():
    """Try to vectorize loop with dynamic shape.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.LoopVectorizeDynamic()  # type: ignore


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
    """EliminateStorageSyncForMBarrier
    """
    return _ffi_api.EliminateStorageSyncForMBarrier()  # type: ignore
