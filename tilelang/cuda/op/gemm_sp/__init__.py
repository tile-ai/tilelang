"""CUDA sparse GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from tilelang.tileop.gemm.registry import load_gemm_impl_class
from tilelang.cuda.target import target_is_cuda, target_is_turing, target_is_volta

GEMM_SP_INST_MMA_SP = "cuda.mma.sp"
GEMM_SP_INST_WGMMA_SP = "cuda.wgmma.sp"
_IMPLEMENTATIONS = {"GemmSPMMA": "gemm_sp_mma", "GemmSPWGMMA": "gemm_sp_wgmma"}


def _implementation(name):
    return f"{__name__}.{_IMPLEMENTATIONS[name]}:{name}"


def __getattr__(name):
    if name not in _IMPLEMENTATIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = load_gemm_impl_class(_implementation(name))
    globals()[name] = implementation
    return implementation


def __dir__():
    return sorted(set(globals()) | _IMPLEMENTATIONS.keys())


def _match_mma(target) -> bool:
    return target_is_cuda(target) and not (target_is_volta(target) or target_is_turing(target))


def _match_wgmma(target) -> bool:
    return target_is_cuda(target)


register_gemm_sp_impl("cuda.mma.sp", GEMM_SP_INST_MMA_SP, _match_mma, _implementation("GemmSPMMA"))
register_gemm_sp_impl("cuda.wgmma.sp", GEMM_SP_INST_WGMMA_SP, _match_wgmma, _implementation("GemmSPWGMMA"))

__all__ = tuple(dict.fromkeys([*[name for name in globals() if not name.startswith("_")], *_IMPLEMENTATIONS]))
