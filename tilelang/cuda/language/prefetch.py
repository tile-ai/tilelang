from typing import Literal

from tvm import tirx

from tilelang.language.copy_op import EVICTION_POLICY_IDS
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
    eviction_policy: Literal["evict_normal", "evict_last"] | None = None,
) -> tirx.Call:
    """Prefetch the global or local memory cache line containing ``addr`` into L2.

    Args:
        addr: A global or local buffer element, such as ``x[i]``. Only its address is used.
        eviction_policy: Compile-time cache eviction priority. None leaves the priority unspecified.
            "evict_normal" and "evict_last" require global memory and sm_80 or later.

    Returns:
        A call to the L2 prefetch intrinsic. Prefetch is a performance hint and
        does not guarantee cache residency or synchronize memory accesses.
    """
    if eviction_policy not in (None, "evict_normal", "evict_last"):
        raise ValueError(f"prefetch_l2 eviction_policy expects None, 'evict_normal', or 'evict_last'. Got {eviction_policy!r}")

    if is_local(addr):
        if eviction_policy is not None:
            raise ValueError("prefetch_l2 eviction_policy requires global memory")
    elif not is_global(addr):
        raise ValueError(f"prefetch_l2 requires global or local memory, got scope={addr.buffer.scope()}")

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetch_l2"),
        tirx.address_of(addr),
        -1 if eviction_policy is None else EVICTION_POLICY_IDS[eviction_policy],
        is_global(addr),
    )
