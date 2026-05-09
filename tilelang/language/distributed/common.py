from __future__ import annotations

from tvm import tir


def get_rank():
    """Get the rank of the current process."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))
