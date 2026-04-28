from __future__ import annotations

from tvm.target import Target
from tilelang.backend.registry import (
    allowed_execution_backends_for_target,
    resolve_execution_backend as _resolve_execution_backend,
)


def allowed_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    """Return allowed execution backends for a given TVM target.

    include_unavailable: if False, this will filter out backends that are known
    to be unavailable at runtime (e.g., NVRTC without cuda-python installed).
    """
    return allowed_execution_backends_for_target(target, include_unavailable=include_unavailable)


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    """Resolve an execution backend string to a concrete backend.

    - Supports the alias "dlpack" -> "tvm_ffi".
    - Supports the sentinel "auto" which selects a sensible default per target.
    - Validates the combination (target, backend) and raises with helpful
      alternatives when invalid.
    """
    return _resolve_execution_backend(requested, target)
