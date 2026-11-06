from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from .gemm_amx import GEMM_INST_AMX_BF16, GEMM_INST_AMX_INT8, GemmAMXBF16, GemmAMXInt8


def _match_scalar(target) -> bool:
    return target.kind.name in {"c", "llvm"}


def _match_amx(target) -> bool:
    return target.kind.name == "c" and "amx" in {str(key) for key in target.keys}


register_gemm_impl("cpu.scalar", GEMM_INST_SCALAR, _match_scalar, GemmScalar)
register_gemm_impl("cpu.amx.bf16", GEMM_INST_AMX_BF16, _match_amx, GemmAMXBF16)
register_gemm_impl("cpu.amx.int8", GEMM_INST_AMX_INT8, _match_amx, GemmAMXInt8)
