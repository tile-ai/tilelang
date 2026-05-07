from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_mma import GEMM_INST_MMA, GemmMMA
from tilelang.tileop.gemm.gemm_mma_sm70 import GemmMMASm70
from tilelang.tileop.gemm.gemm_tcgen05 import GEMM_INST_TCGEN05, GemmTCGEN5
from tilelang.tileop.gemm.gemm_wgmma import GEMM_INST_WGMMA, GemmWGMMA
from tilelang.utils.target import target_is_cuda, target_is_volta


def _match_mma(target) -> bool:
    return target_is_cuda(target) and not target_is_volta(target)


def _match_mma_sm70(target) -> bool:
    return target_is_volta(target)


def _match_wgmma(target) -> bool:
    return target_is_cuda(target)


def _match_tcgen05(target) -> bool:
    return target_is_cuda(target)


register_gemm_impl("cuda.mma", GEMM_INST_MMA, _match_mma, GemmMMA)
register_gemm_impl("cuda.mma_sm70", GEMM_INST_MMA, _match_mma_sm70, GemmMMASm70)
register_gemm_impl("cuda.wgmma", GEMM_INST_WGMMA, _match_wgmma, GemmWGMMA)
register_gemm_impl("cuda.tcgen05", GEMM_INST_TCGEN05, _match_tcgen05, GemmTCGEN5)
