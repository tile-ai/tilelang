from __future__ import annotations

from collections.abc import Iterable

from tvm.target import Target
from tilelang.jit.adapter.utils import is_cutedsl_target
from tilelang.env import env as _env

# Canonical names for execution backends used internally
_CANONICAL_MAP = {
    "dlpack": "tvm_ffi",  # historical alias
}


def _canon_backend(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).lower()
    return _CANONICAL_MAP.get(key, key)


def _target_kind(target: Target) -> str:
    # tvm.target.Target always has kind.name
    return target.kind.name


def allowed_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    """Return allowed execution backends for a given TVM target kind.

    include_unavailable: if False, this will filter out backends that are known
    to be unavailable at runtime (e.g., NVRTC without cuda-python installed).
    """
    kind = _target_kind(target)

    if is_cutedsl_target(target):
        return ["cutedsl"]
    elif kind == "cuda":
        allowed = ["tvm_ffi", "nvrtc", "cython"]
    elif kind == "hip":
        allowed = ["tvm_ffi", "cython"]
    elif kind == "metal":
        allowed = ["torch"]
    elif kind == "c":  # CPU C backend
        allowed = ["cython", "tvm_ffi"]
    else:
        # Fallback: prefer portable hosts
        allowed = ["cython", "tvm_ffi"]

    if not include_unavailable:
        # Drop NVRTC if not importable
        try:
            from tilelang.jit.adapter.nvrtc import is_nvrtc_available  # lazy

            if not is_nvrtc_available and "nvrtc" in allowed:
                allowed = [b for b in allowed if b != "nvrtc"]
        except Exception:
            # Be conservative and keep nvrtc if detection itself fails
            pass

    return allowed


def _format_options(options: Iterable[str]) -> str:
    return ", ".join(sorted(options))


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    """Resolve an execution backend string to a concrete backend.

    - Supports the alias "dlpack" -> "tvm_ffi".
    - Supports the sentinel "auto" which selects a sensible default per target.
    - Validates the combination (target, backend) and raises with helpful
      alternatives when invalid.
    """
    req = _canon_backend(requested)
    allowed_all = allowed_backends_for_target(target, include_unavailable=True)
    allowed_avail = allowed_backends_for_target(target, include_unavailable=False)

    def _require_gemm_v1_for_cutedsl():
        if not _env.use_gemm_v1():
            raise ValueError(
                "CuTeDSL backend requires GEMM v1. Please set environment variable TILELANG_USE_GEMM_V1=1 before importing tilelang."
            )
        # Fail fast with a clear error if CuTeDSL dependencies are missing or incompatible.
        try:
            from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available  # lazy

            check_cutedsl_available()
        except ImportError as e:
            # Keep resolve_execution_backend's error semantics (ValueError) while
            # preserving the actionable ImportError message.
            raise ValueError(str(e)) from e

    # Default selection for auto/None
    if req in (None, "auto"):
        if is_cutedsl_target(target):
            _require_gemm_v1_for_cutedsl()
            return "cutedsl"
        kind = _target_kind(target)
        if kind == "cuda":
            choice = "tvm_ffi"
        elif kind == "metal":
            choice = "torch"
        else:
            choice = "cython"
        # If the chosen default is not available (very rare), fall back to first available
        if choice not in allowed_avail and allowed_avail:
            choice = allowed_avail[0]
        return choice

    # Validate against allowed
    if req not in allowed_all:
        raise ValueError(
            f"Invalid execution backend '{requested}' for target '{_target_kind(target)}'. "
            f"Allowed: {_format_options(allowed_all)}. Tip: use execution_backend='auto'."
        )

    # Promote to availability-aware set for nicer errors (e.g., nvrtc not installed)
    if req not in allowed_avail:
        raise ValueError(
            f"Execution backend '{requested}' requires extra dependencies and is not available now. "
            f"Try one of: {_format_options(allowed_avail)}."
        )

    # CuTeDSL requires GEMM v1
    if req == "cutedsl":
        _require_gemm_v1_for_cutedsl()

    return req
