from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.tileop.gemm.gemm_metal import GEMM_INST_METAL, GEMM_INST_METAL_COOPERATIVE_TENSOR, GemmMetal, GemmMetalSimdGroup
from tilelang.utils.target import target_is_metal


def _match_metal(target) -> bool:
    return target_is_metal(target)


register_gemm_impl(GEMM_INST_METAL, GEMM_INST_METAL, _match_metal, GemmMetalSimdGroup)
register_gemm_impl(GEMM_INST_METAL_COOPERATIVE_TENSOR, GEMM_INST_METAL_COOPERATIVE_TENSOR, _match_metal, GemmMetal)
