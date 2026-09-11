"""CUDA dialect of the copy operators: the common ops plus CUDA copy hints."""

from __future__ import annotations

from typing import Any, Literal

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import (
    EVICTION_POLICY_IDS,
    copy as _common_copy,
    im2col_impl,
)

__all__ = ["copy", "im2col"]


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions, with CUDA lowering hints.

    Same semantics as the common :func:`tilelang.language.copy_op.copy`; the
    extra keywords steer how the CUDA backend lowers the copy. They are
    performance hints recorded on the tile op: compiling the same kernel for a
    target that has no use for them leaves the result unchanged.

    Args:
        src: Source memory region (Buffer, BufferLoad or BufferRegion).
        dst: Destination memory region.
        coalesced_width (Optional[int], keyword-only): Width for coalesced
            memory access. Defaults to None.
        disable_tma (bool, keyword-only): Never lower this copy through TMA
            even when the shape and scopes qualify. Defaults to False.
        eviction_policy (Optional[str], keyword-only): L2 cache eviction
            priority for the generated load/store or TMA instruction, one of
            ``"evict_normal"``, ``"evict_first"``, ``"evict_last"``.
        prefer_instruction (Optional[str], keyword-only): Preferred lowering
            instruction category: ``"tma"``, ``"cp_async"`` or ``"sync"``. For
            ``"tma"``, T.copy keeps synchronous copy semantics; global ->
            shared copies lower through TMA with an automatically allocated
            barrier and wait when constraints are satisfied.
        annotations (Optional[dict], keyword-only): Additional annotations
            dict; values in it take precedence over the individual keywords.
        loop_layout (Optional[Fragment], keyword-only): Parallel loop layout
            hint for the SIMT copy path.

    Returns:
        tirx.Call: A handle to the copy operation.
    """
    ann: dict = dict(annotations) if annotations is not None else {}
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        ann["eviction_policy"] = EVICTION_POLICY_IDS[eviction_policy]
    if "prefer_instruction" not in ann and prefer_instruction is not None:
        ann["prefer_instruction"] = tirx.StringImm(prefer_instruction)
    return _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        annotations=ann or None,
        loop_layout=loop_layout,
    )


def im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tirx.PrimExpr,
    c_step: tirx.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Perform im2col transformation for 2D convolution, with CUDA hints.

    Same semantics as the common :func:`tilelang.language.copy_op.im2col`;
    ``eviction_policy`` is the L2 cache hint consumed by the CUDA TMA im2col
    lowering (ignored by the generic SIMT fallback other targets use).
    """
    return im2col_impl(img, col, nhw_step, c_step, kernel, stride, dilation, pad, eviction_policy=eviction_policy, annotations=annotations)
