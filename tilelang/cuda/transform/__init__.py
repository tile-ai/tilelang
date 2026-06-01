"""CUDA-specific transformation frontends."""

from .. import _ffi_api


def ProducerConsumerWarpSpecialized():
    """Producer-consumer warp specialization at the tile-op level."""
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def ProducerConsumerWarpSpecializedTiled():
    """Compatibility alias for ProducerConsumerWarpSpecialized."""
    return ProducerConsumerWarpSpecialized()


def LowerBlackwell2SM():
    """Lower Blackwell 2-SM TCGEN05 annotations."""
    return _ffi_api.LowerBlackwell2SM()  # type: ignore


def LowerHopperIntrin():
    """LowerHopperIntrin"""
    if hasattr(_ffi_api, "LowerHopperIntrin"):
        return _ffi_api.LowerHopperIntrin()  # type: ignore
    return lambda f: f


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def LowerSharedTmem():
    """Lower CUDA shared.tmem buffers."""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LowerSharedBarrier():
    """Lower CUDA shared.barrier buffers."""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def FuseMBarrierArriveExpectTx():
    """Fuse CUDA mbarrier expect_tx/TMA/arrive sequences."""
    return _ffi_api.FuseMBarrierArriveExpectTx()  # type: ignore


def LowerLDGSTG():
    """Lower CUDA global vector loads/stores to ldg/stg intrinsics."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


def MarkCudaSyncCalls(have_pdl: bool = False):
    """Mark CUDA PDL synchronization calls."""
    return _ffi_api.MarkCudaSyncCalls(have_pdl)  # type: ignore


def InjectFenceProxy():
    """Inject CUDA async/generic proxy fences."""
    return _ffi_api.InjectFenceProxy()  # type: ignore


def InjectTcgen05Fence():
    """Inject CUDA TCGEN05/TMEM synchronization fences."""
    return _ffi_api.InjectTcgen05Fence()  # type: ignore


def AnnotateWarpGroupRegAlloc():
    """Annotate CUDA warp-group register allocation."""
    return _ffi_api.AnnotateWarpGroupRegAlloc()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "AnnotateWarpGroupRegAlloc",
    "FuseMBarrierArriveExpectTx",
    "InjectFenceProxy",
    "InjectTcgen05Fence",
    "LowerBlackwell2SM",
    "LowerHopperIntrin",
    "LowerLDGSTG",
    "LowerL2Persistent",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "MarkCudaSyncCalls",
    "PersistThreadblock",
    "ProducerConsumerWarpSpecialized",
    "ProducerConsumerWarpSpecializedTiled",
]
