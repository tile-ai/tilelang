from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_metal import GEMM_INST_METAL, GemmMetal
from tilelang.utils.target import target_is_metal


def _match_metal(target) -> bool:
    return target_is_metal(target)


register_gemm_impl("metal.simdgroup", GEMM_INST_METAL, _match_metal, GemmMetal)
