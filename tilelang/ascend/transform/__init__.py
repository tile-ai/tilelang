"""Ascend-only TIR transform passes."""
# pylint: disable=invalid-name

from tilelang.transform import _ffi_api

from . import z3_scheduler  # noqa: F401


def NormalizeGemm():
    """Downgrade L1-input GEMMs to L0A/L0B GEMMs ahead of layout inference.

    Materializes the ``ascend_gemm_l1`` sub-K pipeline as ordinary tile ops, so
    layout inference, dependency analysis, AutoSchedule and InsertSync all see the
    L1->L0A/L0B staging copies and the MAD instead of one opaque call.
    """
    return _ffi_api.NormalizeGemm()  # type: ignore


def InsertNd2Nz():
    """Rewrite layout-driven UB ND->UB NZ and UB->L1 ND->NZ copies."""
    return _ffi_api.InsertNd2Nz()  # type: ignore


def NormalizeControlFlowForSchedule():
    """Normalize control flow ahead of AutoSchedule.

    Rewrites each ``while cond: BODY`` into a bounded serial ``for`` loop
    (tagged ``synthetic_while``) so AutoSchedule can pipeline the body, then
    extracts complex ``if`` conditions and buffer-dependent serial/unrolled
    loop bounds into temporary ``Bind`` variables.
    :func:`RestoreWhileLoops` undoes the while-rewrite after scheduling.
    """
    return _ffi_api.NormalizeControlFlowForSchedule()  # type: ignore


def RestoreWhileLoops():
    """Rewrite ``synthetic_while``-tagged for loops back into ``while(true)``."""
    return _ffi_api.RestoreWhileLoops()  # type: ignore


def AnnotateMultiBufferEligible():
    """Annotate scheduled For loops with buffers that can be multi-buffered.

    Requires :func:`NormalizeControlFlowForSchedule` followed by
    :func:`MaterializeScheduleUnits`, so buffer-dependent control expressions
    are schedulable tasks and manual stages and flattened scheduling guards are
    available through the shared IRStructure codec.
    """
    return _ffi_api.AnnotateMultiBufferEligible()  # type: ignore


def NormalizeConflictHints():
    """Normalize ``T.assume_no_conflict`` and ``T.assume_conflict`` markers."""
    return _ffi_api.NormalizeConflictHints()  # type: ignore


def EstimateLatency():
    """Annotate every Ascend AutoSchedule task with latency and II."""
    return _ffi_api.EstimateLatency()  # type: ignore


def MaterializeScheduleUnits():
    """Normalize task boundaries and materialize trivial schedule units."""
    return _ffi_api.MaterializeScheduleUnits()  # type: ignore


def AscendRemoveNoOp():
    """Remove no-op statements using equality-safe store analysis."""
    return _ffi_api.AscendRemoveNoOp()  # type: ignore


def LegalizeSimdMerging():
    """Lower MODE_MERGING SIMD assignments to explicit read-write calls."""
    return _ffi_api.LegalizeSimdMerging()  # type: ignore


def UnrollLoopSkipVF():
    """Expand explicit unroll loops outside SIMD_VF / SIMT_VF blocks.

    This pre-AutoSchedule pass only materializes ``T.unroll(..., explicit=True)``
    loops. Non-explicit unroll hints, serial loops, and all VF block interiors
    are preserved for their later lowering stages.
    """
    return _ffi_api.UnrollLoopSkipVF()  # type: ignore


def AutoSchedule():
    """Schedule materialized internal schedule units for downstream lowering."""
    return _ffi_api.AutoSchedule()  # type: ignore


def AssignCore():
    """Annotate each scheduled task with its widest legal AIV/AIC mask."""
    return _ffi_api.AssignCore()  # type: ignore


def PrepareMultiBuffer():
    """Choose multi-buffer clocks and add counter init/advance tasks."""
    return _ffi_api.PrepareMultiBuffer()  # type: ignore


def ResolveCore():
    """Narrow core candidates from actual scalar and counter consumers."""
    return _ffi_api.ResolveCore()  # type: ignore


def InsertSync():
    """Insert synchronization into scheduled Ascend TIR."""
    return _ffi_api.InsertSync()  # type: ignore


def MaterializeMultiBuffer():
    """Lower logical multi-buffer accesses to physical buffer versions."""
    return _ffi_api.MaterializeMultiBuffer()  # type: ignore


def LowerScheduledTIR():
    """Lower scheduled TIR into final AIV/AIC core bodies."""
    return _ffi_api.LowerScheduledTIR()  # type: ignore


def RewriteDualCopy():
    """Lower dual-copy operations using the enclosing vector-core sid."""
    return _ffi_api.RewriteDualCopy()  # type: ignore


def NormalizeBufferVersion():
    """Lower staged manual/automatic versions to scoped internal attributes."""
    return _ffi_api.NormalizeBufferVersion()  # type: ignore


def RewriteAscendBufferVersionLayout():
    """Align each physical version of a multi-buffered UB allocation to 32 bytes."""
    return _ffi_api.RewriteAscendBufferVersionLayout()  # type: ignore


def NormalizeAscendFractalStorage():
    """Normalize Cube storage and aliases before OOB padding and scheduling."""
    return _ffi_api.NormalizeAscendFractalStorage()  # type: ignore


def AscendSimdVFLowerParallel():
    """Lower T.Parallel inside SIMD_VF blocks to MicroAPI register-level vector ops."""
    return _ffi_api.AscendSimdVFLowerParallel()  # type: ignore


def AscendThreadSync(storage_scope: str):
    """Insert thread-storage synchronization independently within each SIMT_VF."""
    return _ffi_api.AscendThreadSync(storage_scope)


def AscendLayoutInference():
    """Ascend fork of LayoutInference: VF regions are opaque to the worklist
    and SIMT_VF bodies infer against the region's own lane scope."""
    return _ffi_api.AscendLayoutInference()  # type: ignore


def AscendLowerTileOp():
    """Ascend fork of LowerTileOp: VF region scopes, buffer-version key remap,
    and no CUDA async-copy post-processing."""
    return _ffi_api.AscendLowerTileOp()  # type: ignore


def AscendInsertOOBPadding():
    """Clamp DMA copy OOB tails and emit GM->L1 padding fills before AutoSchedule.

    For every supported DMA copy this rewrites the copy to its in-bounds ranges;
    for a padded GM->L1 copy it also appends semantic T.fill operations with
    exact destination regions. Runs after InsertNd2Nz and before AutoSchedule
    so each fill is scheduled as its own MTE2 task and ordered against L1->L0
    consumers. LowerTileOp later converts the fills to ascend_fill_l1, which
    codegen emits as asc_fill_l1.
    """
    return _ffi_api.AscendInsertOOBPadding()  # type: ignore


def InferBufferAliases():
    """Infer which on-chip buffers in a manually scheduled kernel may reuse storage.

    The resulting pairwise compatibility contract is consumed by
    :func:`MergeUBAllocations`.
    """
    return _ffi_api.InferBufferAliases()  # type: ignore


def MergeUBAllocations(align_bytes: int = 16, disable_reuse: bool = False):
    """Merge Ascend on-chip allocations from a pairwise reuse contract.

    AutoSchedule writes this contract during synchronization insertion. Manual
    schedules must run :func:`InferBufferAliases` first.

    Buffer reuse is always performed unless ``disable_reuse`` is set.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeUBAllocations(align_bytes, disable_reuse)  # type: ignore


def MarkScalarDcacheBypass():
    """Rewrite scalar global buffer accesses to use dcache bypass intrinsics
    for buffers that have writes (scalar BufferStore or MTE copy). Pure-read
    buffers keep normal BufferLoad (dcache path).
    """
    return _ffi_api.MarkScalarDcacheBypass()  # type: ignore


def RewriteFp4ToFp4x2():
    """Retype genuine fp4 (float4_e2m1fn, lanes=1) storage into the 1-byte
    packed-pair form (float4_e2m1fnx2, lanes=2), halving fp4 element counts /
    offsets / indices so pointer arithmetic addresses bytes correctly. fp4
    views over non-fp4 storage are left untouched.
    """
    return _ffi_api.RewriteFp4ToFp4x2()  # type: ignore


def RewriteFlagToBuf():
    """Normalize each hard_event's synchronization into its 8-slot flag namespace.
    Sparse flag_ids that extend outside [0,8) and occupy at most 8 slots are
    compacted into [0,8) without spilling. When more than 8 slots are required,
    a 0/1 knapsack (capacity 8) keeps the subset of sync-point blocks that fills
    the flag slots best (renumbered into [0,8)); the rest spill to the shared
    32-slot get_buf/rls_buf mutex pool. Hard_events whose flag_ids are already
    within [0,8) are untouched.

    Must run after InferBufferAliases for manual schedules because alias
    inference uses set_flag/wait_flag as its liveness-graph anchors. The
    standard pipeline places this after MergeUBAllocations.
    """
    return _ffi_api.RewriteFlagToBuf()  # type: ignore


__all__ = [
    "NormalizeAscendFractalStorage",
    "AnnotateMultiBufferEligible",
    "AscendRemoveNoOp",
    "AscendInsertOOBPadding",
    "AscendSimdVFLowerParallel",
    "AscendThreadSync",
    "AscendLayoutInference",
    "AscendLowerTileOp",
    "AssignCore",
    "AutoSchedule",
    "InsertSync",
    "LowerScheduledTIR",
    "EstimateLatency",
    "InferBufferAliases",
    "InsertNd2Nz",
    "LegalizeSimdMerging",
    "MarkScalarDcacheBypass",
    "MaterializeMultiBuffer",
    "MergeUBAllocations",
    "NormalizeBufferVersion",
    "NormalizeControlFlowForSchedule",
    "NormalizeConflictHints",
    "NormalizeGemm",
    "PrepareMultiBuffer",
    "ResolveCore",
    "RewriteAscendBufferVersionLayout",
    "RewriteDualCopy",
    "RestoreWhileLoops",
    "RewriteFp4ToFp4x2",
    "RewriteFlagToBuf",
    "UnrollLoopSkipVF",
]
