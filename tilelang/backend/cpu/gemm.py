from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar


def _match_scalar(target) -> bool:
    return target.kind.name in {"c", "llvm"}


register_gemm_impl("cpu.scalar", GEMM_INST_SCALAR, _match_scalar, GemmScalar)
