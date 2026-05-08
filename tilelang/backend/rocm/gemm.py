from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_mfma import GEMM_INST_MFMA, GemmMFMA
from tilelang.tileop.gemm.gemm_wmma import GEMM_INST_WMMA, GemmWMMA
from tilelang.utils.target import target_is_hip


def _match_mfma(target) -> bool:
    return target_is_hip(target)


def _match_wmma(target) -> bool:
    return target_is_hip(target)


register_gemm_impl("rocm.mfma", GEMM_INST_MFMA, _match_mfma, GemmMFMA)
register_gemm_impl("rocm.wmma", GEMM_INST_WMMA, _match_wmma, GemmWMMA)
