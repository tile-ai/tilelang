"""TileIR load/store hints on the common copy operation."""

from __future__ import annotations

from typing import Any, Literal

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import EVICTION_POLICY_IDS, copy as _common_copy

__all__ = ["copy"]


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    latency: int | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy memory with optional TileIR load/store optimization hints.

    ``latency`` is a traffic-intensity hint in [1, 10]; higher values request
    deeper prefetch. ``disable_tma=True`` disallows TMA for this copy. Unset
    hints let the compiler choose. Values in ``annotations`` take precedence
    over keyword arguments. Cache eviction and SIMT layout hints are recorded
    for compatibility but are not consumed by TileIR lowering.
    """
    ann: dict = dict(annotations) if annotations is not None else {}
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "tileir.latency" not in ann and latency is not None:
        ann["tileir.latency"] = latency
    if "eviction_policy" not in ann and eviction_policy is not None:
        ann["eviction_policy"] = EVICTION_POLICY_IDS[eviction_policy]
    return _common_copy(src, dst, coalesced_width=coalesced_width, annotations=ann or None, loop_layout=loop_layout)
