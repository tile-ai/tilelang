"""Ascend GEMM op registrations."""

from __future__ import annotations

from tilelang.ascend.target import target_is_ascend
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_mad import GEMM_INST_MAD, GemmMAD
from .gemm_mad_blockscaled import GEMM_INST_MAD_BLOCK_SCALED, GemmMADBlockScaled


register_gemm_impl("ascend.mad", GEMM_INST_MAD, target_is_ascend, GemmMAD)
register_gemm_impl(
    "ascend.mad.blockscaled",
    GEMM_INST_MAD_BLOCK_SCALED,
    target_is_ascend,
    GemmMADBlockScaled,
)
