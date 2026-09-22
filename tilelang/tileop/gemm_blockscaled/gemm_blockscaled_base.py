"""Shared accessors for block-scaled GEMM backend implementations."""

from __future__ import annotations

from tvm import tirx
from tvm.ir import PrimExpr


class GemmBlockScaledMixin:
    """Scale-factor operands and knobs of a block-scaled GEMM implementation.

    Mixed into a dense backend implementation class (``GemmTCGEN5``,
    ``GemmMMA``, ...) whose ``gemm_node`` is a ``GemmBlockScaled``. It owns
    everything the dense classes must not know about: the SFA/SFB regions,
    the logical K-axis start and the ``sf_*`` annotations. Which scale
    layouts a backend supports stays with that backend.
    """

    gemm_node: object

    @property
    def SFARegion(self) -> tirx.BufferRegion:
        return self.gemm_node.sfaRegion

    @property
    def SFBRegion(self) -> tirx.BufferRegion:
        return self.gemm_node.sfbRegion

    @property
    def sf_k_start(self) -> PrimExpr:
        return self.gemm_node.sfKStart

    @property
    def sf_a_granularity_k(self) -> int:
        return self._sf_granularity_k("sf_a_granularity_k")

    @property
    def sf_b_granularity_k(self) -> int:
        return self._sf_granularity_k("sf_b_granularity_k")

    @property
    def sf_layout(self) -> str:
        """Scale-factor layout name from the annotations; ``"rowmajor"`` by default."""
        layout = self.gemm_node.annotations.get("sf_layout", "rowmajor")
        if isinstance(layout, tirx.StringImm):
            layout = layout.value
        return str(layout)

    def _sf_granularity_k(self, key: str) -> int:
        value = self.gemm_node.annotations.get(key)
        if value is None:
            raise ValueError(f"Block-scaled GEMM requires the {key} annotation (K elements covered by one scale factor)")
        return int(value)
