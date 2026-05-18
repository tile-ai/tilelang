"""CUDA sparse GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from .gemm_sp_mma import GemmSPMMA
from tilelang.utils.target import target_is_cuda


register_gemm_sp_impl("cuda.GemmSPMMA", target_is_cuda, GemmSPMMA)
