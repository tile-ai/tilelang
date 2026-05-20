"""CUDA sparse GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from tilelang.cuda.op.gemm_sp.gemm_sp_mma import GEMM_SP_INST_MMA_SP, GemmSPMMA
from tilelang.cuda.op.gemm_sp.gemm_sp_wgmma import GEMM_SP_INST_WGMMA_SP, GemmSPWGMMA
from tilelang.utils.target import target_is_cuda, target_is_turing, target_is_volta


def _match_mma(target) -> bool:
    return target_is_cuda(target) and not (target_is_volta(target) or target_is_turing(target))


def _match_wgmma(target) -> bool:
    return target_is_cuda(target)


register_gemm_sp_impl("cuda.mma.sp", GEMM_SP_INST_MMA_SP, _match_mma, GemmSPMMA)
register_gemm_sp_impl("cuda.wgmma.sp", GEMM_SP_INST_WGMMA_SP, _match_wgmma, GemmSPWGMMA)
