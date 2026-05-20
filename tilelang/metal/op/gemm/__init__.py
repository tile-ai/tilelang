from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.utils.target import target_is_metal
from .gemm_metal_scalar import GEMM_INST_METAL_SCALAR, GemmMetalScalar


def _match_metal_scalar(target) -> bool:
    return target_is_metal(target)


register_gemm_impl(
    "metal.scalar",
    GEMM_INST_METAL_SCALAR,
    _match_metal_scalar,
    GemmMetalScalar,
)
