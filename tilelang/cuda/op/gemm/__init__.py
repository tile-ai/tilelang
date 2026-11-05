"""CUDA GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm.registry import load_gemm_impl_class, register_gemm_impl
from tilelang.cuda.target import target_is_cuda, target_is_sm120, target_is_turing, target_is_volta

GEMM_INST_FMA = "cuda.fma"
GEMM_INST_MMA = "cuda.mma"
GEMM_INST_MMA_BLOCK_SCALED = "cuda.mma.blockscaled"
GEMM_INST_TCGEN05 = "cuda.tcgen05"
GEMM_INST_TCGEN05_BLOCK_SCALED = "cuda.tcgen05.blockscaled"
GEMM_INST_WGMMA = "cuda.wgmma"

_IMPLEMENTATIONS = {
    "GemmFMA": "gemm_fma",
    "GemmMMA": "gemm_mma",
    "GemmMMASm70": "gemm_mma_sm70",
    "GemmMMASm75": "gemm_mma_sm75",
    "GemmMMASm120BlockScaled": "gemm_mma_sm120",
    "GemmTCGEN5": "gemm_tcgen05",
    "GemmTCGEN5BlockScaled": "gemm_tcgen05_blockscaled",
    "GemmWGMMA": "gemm_wgmma",
}


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


def _match_mma_sm70(target) -> bool:
    return target_is_volta(target)


def _match_mma_sm75(target) -> bool:
    return target_is_turing(target)


def _match_mma_sm120(target) -> bool:
    return target_is_sm120(target)


def _match_fma(target) -> bool:
    # _match_fma checks CUDA FMA capability; SelectInst decides when to use it.
    return target_is_cuda(target)


def _match_wgmma(target) -> bool:
    return target_is_cuda(target)


def _match_tcgen05(target) -> bool:
    return target_is_cuda(target)


register_gemm_impl("cuda.mma", GEMM_INST_MMA, _match_mma, _implementation("GemmMMA"))
register_gemm_impl("cuda.mma_sm70", GEMM_INST_MMA, _match_mma_sm70, _implementation("GemmMMASm70"))
register_gemm_impl("cuda.mma_sm75", GEMM_INST_MMA, _match_mma_sm75, _implementation("GemmMMASm75"))
register_gemm_impl(
    "cuda.mma_sm120",
    GEMM_INST_MMA_BLOCK_SCALED,
    _match_mma_sm120,
    _implementation("GemmMMASm120BlockScaled"),
)
register_gemm_impl("cuda.fma", GEMM_INST_FMA, _match_fma, _implementation("GemmFMA"))
register_gemm_impl("cuda.wgmma", GEMM_INST_WGMMA, _match_wgmma, _implementation("GemmWGMMA"))
register_gemm_impl("cuda.tcgen05", GEMM_INST_TCGEN05, _match_tcgen05, _implementation("GemmTCGEN5"))
register_gemm_impl(
    "cuda.tcgen05_blockscaled",
    GEMM_INST_TCGEN05_BLOCK_SCALED,
    _match_tcgen05,
    _implementation("GemmTCGEN5BlockScaled"),
)

__all__ = tuple(dict.fromkeys([*[name for name in globals() if not name.startswith("_")], *_IMPLEMENTATIONS]))
