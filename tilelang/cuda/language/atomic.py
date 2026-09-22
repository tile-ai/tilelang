"""CUDA dialect of the atomic operators: the common ops plus CUDA knobs."""

from __future__ import annotations

from tvm.tirx import Buffer, PrimExpr

from tilelang.language.atomic import atomic_add as _common_atomic_add

__all__ = ["atomic_add"]


def atomic_add(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False,
    use_tma: bool = False,
    annotations: dict | None = None,
) -> PrimExpr:
    """Atomically add ``value`` into ``dst``, with CUDA lowering knobs.

    Same semantics as the common :func:`tilelang.language.atomic.atomic_add`.
    ``use_tma`` selects the sm90+ TMA ``cp.reduce`` lowering for the
    tile-region path; targets without TMA reject it at compile time.

    Parameters:
        dst (Buffer): Destination buffer/address to apply the atomic add.
        value (PrimExpr): Value to add atomically.
        memory_order (Optional[str]): Memory-order name controlling the atomic
            operation's ordering ("relaxed", "consume", "acquire", "release",
            "acq_rel", "seq_cst").
        return_prev (bool): Return the previous value (scalar path only).
        use_tma (bool): If True, lower the tile-region atomic add through TMA
            ``cp.reduce``. Available on sm90+ only (default False).
        annotations (Optional[dict]): Extra annotations for the tile-region
            path; values in it take precedence over the individual keywords.

    Returns:
        PrimExpr: A handle to the atomic operation.
    """
    ann: dict = dict(annotations) if annotations is not None else {}
    if use_tma:
        ann.setdefault("use_tma", 1)
    return _common_atomic_add(
        dst,
        value,
        memory_order=memory_order,
        return_prev=return_prev,
        annotations=ann or None,
    )
