from tvm import tirx

from tilelang.language.cache import EvictPriority
from tilelang.utils.language import is_global, is_local

__all__ = ["prefetch_l1", "prefetchu_l1", "prefetch_l2"]


def prefetch_l1(addr: tirx.BufferLoad) -> tirx.Call:
    """Prefetch the global or local memory cache line containing ``addr`` into L1."""
    if not is_local(addr) and not is_global(addr):
        raise ValueError(f"prefetch_l1 requires global or local memory, got scope={addr.buffer.scope()}")
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetch_l1"),
        tirx.address_of(addr),
        is_global(addr),
    )


def prefetchu_l1(addr: tirx.BufferLoad) -> tirx.Call:
    """Prefetch the memory cache line containing ``addr`` into uniform L1."""
    if not is_global(addr):
        raise ValueError(f"prefetchu_l1 requires global memory, got scope={addr.buffer.scope()}")
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetchu_l1"),
        tirx.address_of(addr),
    )


def prefetch_l2(
    addr: tirx.BufferLoad,
    eviction_priority: EvictPriority = EvictPriority.Default,
) -> tirx.Call:
    """Prefetch the global or local memory cache line containing ``addr`` into L2.

    Args:
        addr: A global or local buffer element, such as ``x[i]``. Only its address is used.
        eviction_priority: Compile-time eviction priority. Default leaves the priority unspecified.
            EvictNormal and EvictLast require global memory and sm_80 or later.

    Returns:
        A call to the L2 prefetch intrinsic. Prefetch is a performance hint and
        does not guarantee cache residency or synchronize memory accesses.
    """
    if not isinstance(eviction_priority, EvictPriority):
        raise ValueError(f"prefetch_l2 eviction_priority expects EvictPriority. Got {type(eviction_priority)}: {eviction_priority}")
    allowed = (EvictPriority.Default, EvictPriority.EvictNormal, EvictPriority.EvictLast)
    if eviction_priority not in allowed:
        raise ValueError(f"prefetch_l2 eviction_priority expects one of {allowed}. Got {eviction_priority!r}")

    if is_local(addr):
        if eviction_priority != EvictPriority.Default:
            raise ValueError("prefetch_l2 eviction_priority requires global memory")
    elif not is_global(addr):
        raise ValueError(f"prefetch_l2 requires global or local memory, got scope={addr.buffer.scope()}")

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetch_l2"),
        tirx.address_of(addr),
        eviction_priority.value,
        is_global(addr),
    )
