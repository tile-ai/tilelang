from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.metal.target import target_is_metal
from .gemm_metal import GEMM_INST_METAL, GemmMetal


def _match_metal(target) -> bool:
    return target_is_metal(target)


register_gemm_impl("metal.simdgroup", GEMM_INST_METAL, _match_metal, GemmMetal)
