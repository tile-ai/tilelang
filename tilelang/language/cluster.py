from tvm import tir

__all__ = [
    "cluster_arrive_relaxed",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync",
    "block_rank_in_cluster",
]


def cluster_arrive_relaxed() -> tir.PrimExpr:
    """Issue barrier.cluster.arrive.relaxed.aligned."""
    return tir.call_intrin("void", tir.op.Op.get("tl.cluster_arrive_relaxed"))


def cluster_arrive() -> tir.PrimExpr:
    """Issue barrier.cluster.arrive.aligned."""
    return tir.call_intrin("void", tir.op.Op.get("tl.cluster_arrive"))


def cluster_wait() -> tir.PrimExpr:
    """Issue barrier.cluster.wait.aligned."""
    return tir.call_intrin("void", tir.op.Op.get("tl.cluster_wait"))


def cluster_sync() -> tir.PrimExpr:
    """Issue cluster barrier arrive + wait (full synchronization)."""
    return tir.call_intrin("void", tir.op.Op.get("tl.cluster_sync"))


def block_rank_in_cluster() -> tir.PrimExpr:
    """Return the 1-D rank of the calling CTA within its cluster (%%cluster_ctarank)."""
    return tir.call_intrin("uint32", tir.op.Op.get("tl.block_rank_in_cluster"))
