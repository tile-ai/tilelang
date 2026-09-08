"""Ascend GEMM op registrations."""

from __future__ import annotations

from tilelang.ascend.target import target_is_ascend
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_mad import GEMM_INST_MAD, GemmMAD


register_gemm_impl("ascend.mad", GEMM_INST_MAD, target_is_ascend, GemmMAD)
