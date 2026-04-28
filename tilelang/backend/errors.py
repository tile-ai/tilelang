from __future__ import annotations


class BackendResolutionError(ValueError):
    """Raised when no TileLang backend metadata can resolve a target."""


class BackendCodegenError(ValueError):
    """Raised when resolved backend metadata cannot perform requested codegen."""
