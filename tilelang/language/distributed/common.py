from __future__ import annotations

from tvm import tirx


def get_rank():
    """Get the rank of the current process."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_num_ranks"))
