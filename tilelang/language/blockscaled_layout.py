"""Block-scale layout primitives for TileLang FP8 DSL surfaces.

The first consumer is Apple MXFP8/E8M0, where FP8 data remains e4m3 storage
and a separate uint8 scale byte applies to each 32-value block along the
contracted-K axis. The layout object is intentionally small: it carries the
static metadata that schedulers need, while the scale tensors remain ordinary
TileLang buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilelang.tileop.metal_quant import e8m0_to_float

E8M0_BLOCK_SIZE = 32
E8M0_BLOCK_K32 = "e8m0_block_k32"
E8M0_LAYOUT = "logical_unswizzled_k_axis_blocks"
CONTRACTED_K_AXIS = "contracted_k"


@dataclass(frozen=True)
class BlockScaledLayout:
    """Static block-scale metadata for FP8 scaled matmul-like operators.

    ``scale_dtype="e8m0"`` with ``block_size=32`` describes the MXFP8 layout
    used by Metal Path C:

    * A scale shape is ``(K / 32,)``.
    * B scale shape is ``(N, K / 32)`` for per-row scales.
    * A broadcast B scale shape ``(K / 32,)`` is accepted for local probes.
    * The scale block index is the contracted-K block, ``kb = k // 32``.
    """

    scale_dtype: str = "e8m0"
    axis: str = CONTRACTED_K_AXIS
    block_size: int = E8M0_BLOCK_SIZE
    layout: str = E8M0_LAYOUT
    allow_broadcast_b_scale: bool = True

    def __post_init__(self) -> None:
        if self.scale_dtype != "e8m0":
            raise ValueError(
                "BlockScaledLayout currently supports only scale_dtype='e8m0'"
            )
        if self.axis != CONTRACTED_K_AXIS:
            raise ValueError(
                "BlockScaledLayout currently supports only axis='contracted_k'"
            )
        if int(self.block_size) != E8M0_BLOCK_SIZE:
            raise ValueError("E8M0 block-scale layout requires block_size=32")
        if self.layout != E8M0_LAYOUT:
            raise ValueError(
                "E8M0 block-scale layout must be "
                "'logical_unswizzled_k_axis_blocks'"
            )

    @classmethod
    def e8m0_k32(cls) -> "BlockScaledLayout":
        """Return the canonical logical unswizzled E8M0 K/32 layout."""

        return cls(
            scale_dtype="e8m0",
            axis=CONTRACTED_K_AXIS,
            block_size=E8M0_BLOCK_SIZE,
            layout=E8M0_LAYOUT,
        )

    @property
    def scale_format(self) -> str:
        return E8M0_BLOCK_K32

    @property
    def scale_axis(self) -> str:
        return CONTRACTED_K_AXIS

    def scale_blocks(self, k_extent: int) -> int:
        k_extent = int(k_extent)
        if k_extent <= 0:
            raise ValueError(f"block-scaled K extent must be positive, got {k_extent}")
        if k_extent % E8M0_BLOCK_SIZE != 0:
            raise ValueError(
                f"e8m0_block_k32 requires K divisible by 32, got K={k_extent}"
            )
        return k_extent // E8M0_BLOCK_SIZE

    def a_scale_shape(self, k_extent: int) -> tuple[int]:
        return (self.scale_blocks(k_extent),)

    def b_scale_shape(self, n_extent: int, k_extent: int) -> tuple[int, int]:
        n_extent = int(n_extent)
        if n_extent <= 0:
            raise ValueError(f"block-scaled N extent must be positive, got {n_extent}")
        return (n_extent, self.scale_blocks(k_extent))

    def broadcast_b_scale_shape(self, k_extent: int) -> tuple[int]:
        return (self.scale_blocks(k_extent),)

    def scale_index(self, k: Any):
        """Return the contracted-K scale block index for element ``k``."""

        return k // 32

    def decode(self, byte: Any):
        return e8m0_to_float(byte)

    def validate_scale_shapes(
        self,
        *,
        k_extent: int,
        a_scale_shape: tuple[int, ...],
        b_scale_shape: tuple[int, ...],
        n_extent: int | None = None,
    ) -> None:
        blocks = self.scale_blocks(k_extent)
        if tuple(a_scale_shape) != (blocks,):
            raise ValueError(
                "A_scale for e8m0_block_k32 must have shape "
                f"(K / 32,) == ({blocks},), got {a_scale_shape}"
            )
        if tuple(b_scale_shape) == (blocks,) and self.allow_broadcast_b_scale:
            return
        if n_extent is None:
            raise ValueError(
                "B_scale for e8m0_block_k32 must be broadcast (K / 32,) "
                "or per-row (N, K / 32); n_extent is required to validate "
                f"shape {b_scale_shape}"
            )
        expected = (int(n_extent), blocks)
        if tuple(b_scale_shape) != expected:
            raise ValueError(
                "B_scale for e8m0_block_k32 must have shape "
                f"(N, K / 32) == {expected} or broadcast ({blocks},), "
                f"got {b_scale_shape}"
            )
