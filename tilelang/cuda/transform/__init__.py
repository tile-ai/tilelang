"""CUDA-specific transformation frontends."""

from .. import _ffi_api


def ProducerConsumerWarpSpecialized():
    """Producer-consumer warp specialization at the tile-op level.

    This pass runs before LayoutInference and LowerTileOp. It rewrites
    eligible pipelined tile-op loops into warp-specialized producer and
    consumer branches with explicit barrier synchronization.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def LowerBlackwell2SM():
    """Lower 2SM TCGEN5MMA and related on Blackwell target

    Returns:
        fpass : tvm.transform.Pass
            The result pass
    """
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
    """LowerSharedTmem"""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LowerSharedBarrier():
    """LowerSharedBarrier"""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def FuseMBarrierArriveExpectTx():
    """Fuse simple expect_tx -> TMA issue -> arrive back into arrive_and_expect_tx."""
    return _ffi_api.FuseMBarrierArriveExpectTx()  # type: ignore


def LowerLDGSTG():
    """Lower Ramp-based global memory load/store to ldg/stg intrinsics.

    This pass transforms vectorized global memory loads and stores (using Ramp indices)
    into explicit ldg32/64/128/256 and stg32/64/128/256 intrinsics for better codegen.

    Key behaviors:
    - Converts Ramp-based global BufferLoad to ldg intrinsics
    - Converts Ramp-based global BufferStore to stg intrinsics
    - Supports predicated loads (if_then_else with else=0)
    - Supports predicated stores (if in then case)
    - Skips loads in async scope (will be lowered to cp.async)
    - Only enabled for CUDA targets

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerLDGSTG()  # type: ignore


def LowerPTXAsyncCopy():
    """Lower eligible global->shared copies into PTX `cp.async` on CUDA.

    When enabled (pass config `tl.enable_async_copy`, default True), this pass
    may rewrite plain user-written global->shared `BufferStore` patterns (e.g.
    SIMT copies in `T.Parallel`) into `tir.ptx_cp_async`, and insert
    `tir.ptx_commit_group` + `tir.ptx_wait_group(0)` to preserve synchronous
    semantics for normal stores. If explicit commit/wait intrinsics already
    exist, the pass avoids duplicating them (and may insert a missing commit
    immediately before an existing wait to cover injected `cp.async`).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerPTXAsyncCopy()  # type: ignore


def MarkCudaSyncCalls(have_pdl: bool = False):
    """MarkCudaSyncCalls"""
    return _ffi_api.MarkCudaSyncCalls(have_pdl)  # type: ignore


def InjectFenceProxy():
    """InjectFenceProxy

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectFenceProxy()  # type: ignore


def InjectTcgen05Fence():
    """Inject tcgen05.fence::before_thread_sync / after_thread_sync at
    conservative TCGEN05/TMEM synchronization boundaries on Blackwell
    (SM100+) targets.

    The current pass wraps CTA-wide shared-memory syncs and also inserts
    fences around linear mbarrier wait/use and use/arrive handoff patterns.
    It is intentionally conservative and does not try to infer arbitrary
    barrier protocols.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectTcgen05Fence()  # type: ignore


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
    "LowerPTXAsyncCopy",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "MarkCudaSyncCalls",
    "PersistThreadblock",
    "ProducerConsumerWarpSpecialized",
]
