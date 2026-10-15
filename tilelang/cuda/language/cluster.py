from tvm import tirx
from tvm.tirx import BufferLoad

from tilelang.utils.language import retrieve_ptr

__all__ = [
    "cluster_arrive_relaxed",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync",
    "block_rank_in_cluster",
    "clc_try_cancel",
    "clc_try_cancel_multicast",
    "clc_is_canceled",
    "clc_get_first_ctaid_x",
    "clc_get_first_ctaid_y",
    "clc_get_first_ctaid_z",
]


def _to_ptr(value, access_type: str):
    if isinstance(value, BufferLoad):
        return retrieve_ptr(value, access_type=access_type)
    return retrieve_ptr(value, access_type=access_type)


def cluster_arrive_relaxed() -> tirx.PrimExpr:
    """Issue barrier.cluster.arrive.relaxed.aligned."""
    return tirx.call_intrin("void", tirx.op.Op.get("tl.cluster_arrive_relaxed"))


def cluster_arrive() -> tirx.PrimExpr:
    """Issue barrier.cluster.arrive.aligned."""
    return tirx.call_intrin("void", tirx.op.Op.get("tl.cluster_arrive"))


def cluster_wait() -> tirx.PrimExpr:
    """Issue barrier.cluster.wait.aligned."""
    return tirx.call_intrin("void", tirx.op.Op.get("tl.cluster_wait"))


def cluster_sync() -> tirx.PrimExpr:
    """Issue cluster barrier arrive + wait (full synchronization)."""
    return tirx.call_intrin("void", tirx.op.Op.get("tl.cluster_sync"))


def block_rank_in_cluster() -> tirx.PrimExpr:
    """Return the 1-D rank of the calling CTA within its cluster (%%cluster_ctarank)."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.block_rank_in_cluster"))


def clc_try_cancel(result, mbarrier) -> tirx.PrimExpr:
    """Issue a single-CTA cluster launch control query."""
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.clc_try_cancel"),
        _to_ptr(result, "w"),
        _to_ptr(mbarrier, "rw"),
    )


def clc_try_cancel_multicast(result, mbarrier) -> tirx.PrimExpr:
    """Issue a cluster-wide multicast cluster launch control query."""
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.clc_try_cancel_multicast"),
        _to_ptr(result, "w"),
        _to_ptr(mbarrier, "rw"),
    )


def clc_is_canceled(result) -> tirx.PrimExpr:
    """Return 1 when the CLC query successfully canceled a future launch."""
    return tirx.call_intrin(
        "int32",
        tirx.op.Op.get("tl.clc_is_canceled"),
        _to_ptr(result, "r"),
    )


def clc_get_first_ctaid_x(result) -> tirx.PrimExpr:
    """Return the x coordinate of the first CTA in a successful CLC response."""
    return tirx.call_intrin(
        "uint32",
        tirx.op.Op.get("tl.clc_get_first_ctaid_x"),
        _to_ptr(result, "r"),
    )


def clc_get_first_ctaid_y(result) -> tirx.PrimExpr:
    """Return the y coordinate of the first CTA in a successful CLC response."""
    return tirx.call_intrin(
        "uint32",
        tirx.op.Op.get("tl.clc_get_first_ctaid_y"),
        _to_ptr(result, "r"),
    )


def clc_get_first_ctaid_z(result) -> tirx.PrimExpr:
    """Return the z coordinate of the first CTA in a successful CLC response."""
    return tirx.call_intrin(
        "uint32",
        tirx.op.Op.get("tl.clc_get_first_ctaid_z"),
        _to_ptr(result, "r"),
    )
