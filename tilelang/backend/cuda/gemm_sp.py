from __future__ import annotations

from tilelang.backend.gemm_sp import register_gemm_sp_impl
from tilelang.tileop.gemm_sp.gemm_sp_mma import GemmSPMMA
from tilelang.utils.target import target_is_cuda


register_gemm_sp_impl("cuda.GemmSPMMA", target_is_cuda, GemmSPMMA)
