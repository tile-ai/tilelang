from __future__ import annotations

from tilelang.tileop.gemm.registry import load_gemm_impl_class, register_gemm_impl
from tilelang.rocm.target import target_is_hip

GEMM_INST_MFMA = "rocm.mfma"
GEMM_INST_WMMA = "rocm.wmma"
_IMPLEMENTATIONS = {"GemmMFMA": "gemm_mfma", "GemmWMMA": "gemm_wmma"}


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


def _match_mfma(target) -> bool:
    return target_is_hip(target)


def _match_wmma(target) -> bool:
    return target_is_hip(target)


register_gemm_impl("rocm.mfma", GEMM_INST_MFMA, _match_mfma, _implementation("GemmMFMA"))
register_gemm_impl("rocm.wmma", GEMM_INST_WMMA, _match_wmma, _implementation("GemmWMMA"))

__all__ = tuple(dict.fromkeys([*[name for name in globals() if not name.startswith("_")], *_IMPLEMENTATIONS]))
