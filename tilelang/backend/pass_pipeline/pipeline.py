from __future__ import annotations

from collections.abc import Callable

from tvm import IRModule
from tvm.target import Target

LowerFunc = Callable[[IRModule, Target], IRModule]


class PassPipeline:
    """Lowering pass pipeline for a specific backend.

    Each backend should register its own Pipeline so that the compiler can
    resolve the correct pass sequence from the target at runtime.
    """

    def __init__(self, name: str, lower: LowerFunc):
        self.name = name
        self._lower = lower

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        return self._lower(mod, target)


_PIPELINES: dict[str, PassPipeline] = {}


def register_pipeline(pipeline: PassPipeline) -> PassPipeline:
    """Register a lowering pipeline for a backend.

    The pipeline name should match ``target.kind.name`` (e.g. ``"cuda"``,
    ``"hip"``, ``"c"``, ``"llvm"``).
    """
    _PIPELINES[pipeline.name] = pipeline
    return pipeline


def get_pipeline(name: str) -> PassPipeline:
    """Return the registered Pipeline for *name*."""
    if name not in _PIPELINES:
        raise ValueError(f"No pipeline registered for backend '{name}'. Available backends: {list(_PIPELINES.keys())}")
    return _PIPELINES[name]


def resolve_pipeline(target: Target) -> PassPipeline:
    """Resolve the lowering pipeline from a TVM target."""
    return get_pipeline(target.kind.name)
