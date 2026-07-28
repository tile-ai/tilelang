"""Scheduled TIR preparation for the graph compiler."""

import tilelang
from tilelang import tvm
from tvm import s_tir, tirx


def canonicalize_scheduled_ir(mod: tvm.IRModule) -> tvm.IRModule:
    """Prepare schedule-rule output for a backend-owned lowering pipeline.

    Graph scheduling produces thread-binding loops and may leave reduction
    init blocks behind.  Keep those launch loops target-neutral here; the
    selected backend pipeline materializes them for CUDA, ROCm, or CPU.
    """
    # Schedule rules consume the original TE block structure.  Narrow only
    # after scheduling, before layout inference creates Layout/Fragment objects.
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = s_tir.transform.LowerInitBlock()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    return tilelang.transform.ReserveRootBlock()(mod)
