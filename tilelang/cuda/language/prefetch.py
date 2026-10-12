from typing import Literal

from tvm import tirx

from tilelang.language.copy_op import EVICTION_POLICY_IDS

__all__ = ["prefetch_l2"]


def prefetch_l2(
    addr: tirx.BufferLoad,
    eviction_policy: Literal["evict_normal", "evict_last"] = "evict_normal",
) -> tirx.Call:
    """Prefetch the global memory cache line containing ``addr`` into L2.

    Args:
        addr: A global buffer element, such as ``x[i]``. Only its address is used.
        eviction_policy: Compile-time eviction priority, either ``"evict_normal"``
            or ``"evict_last"``. Requires sm_80 or later.

    Returns:
        A call to the L2 prefetch intrinsic. Prefetch is a performance hint and
        does not guarantee cache residency or synchronize memory accesses.
    """
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetch_l2"),
        tirx.address_of(addr),
        tirx.convert(EVICTION_POLICY_IDS[eviction_policy]),
    )
