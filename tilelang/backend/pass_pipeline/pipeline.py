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
        """Run the pipeline and render source snippets for located errors."""
        try:
            return self._lower(mod, target)
        except Exception as exc:
            # Compiler passes append a machine-readable `--> file:line:col`
            # marker when the relevant IR node carries a span. Keep enrichment
            # at this shared boundary so every pipeline caller gets the same
            # user-facing diagnostic without changing exception semantics.
            from tilelang.errors import enrich_error

            enrich_error(exc)
            raise


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
