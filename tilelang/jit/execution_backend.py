from __future__ import annotations

from tvm.target import Target

from tilelang.backend import resolve_backend
from tilelang.backend.backend import ExecutionBackendSpec

# Canonical names for execution backends used internally
_CANONICAL_MAP = {
    "dlpack": "tvm_ffi",  # historical alias
}


def _canon_backend(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).lower()
    return _CANONICAL_MAP.get(key, key)


def allowed_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    """Return execution backends allowed by the resolved compiler backend.

    include_unavailable: if False, this will filter out backends that are known
    to be unavailable at runtime (e.g., NVRTC without cuda-python installed).
    """
    return list(resolve_backend(target).allowed_execution_backends(target, include_unavailable=include_unavailable))


def resolve_execution_backend_spec(requested: str | None, target: Target) -> ExecutionBackendSpec:
    """Resolve a user request to a concrete execution-backend spec."""

    return resolve_backend(target).resolve_execution_backend(_canon_backend(requested), target)


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    """Resolve an execution backend string to a concrete backend.

    - Supports the alias "dlpack" -> "tvm_ffi".
    - Supports the sentinel "auto" which selects a sensible default per target.
    - Validates the combination (target, backend) and raises with helpful
      alternatives when invalid.
    """
    return resolve_execution_backend_spec(requested, target).name
