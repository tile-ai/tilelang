from __future__ import annotations

from collections.abc import Callable

from tvm import IRModule
from tvm.target import Target

LowerFunc = Callable[[IRModule, Target], IRModule]


class PassPipeline:
    """Lowering pass pipeline for a specific backend.

    Each :class:`BackendModule` declares the pipelines it owns so the compiler
    can select the correct pass sequence from the resolved backend context.
    """

    def __init__(self, name: str, lower: LowerFunc):
        self.name = name
        self._lower = lower

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        """Run the pipeline and render source snippets for located errors."""
        try:
            # Tooling scopes are registered lazily.  Keeping this import at
            # the execution boundary avoids pulling developer tools into the
            # backend package during TileLang initialization.
            from tilelang.utils.pass_events import pass_pipeline

            with pass_pipeline(self.name):
                return self._lower(mod, target)
        except Exception as exc:
            # Compiler passes append a machine-readable `--> file:line:col`
            # marker when the relevant IR node carries a span. Keep enrichment
            # at this shared boundary so every pipeline caller gets the same
            # user-facing diagnostic without changing exception semantics.
            from tilelang.errors import enrich_error

            enrich_error(exc)
            raise
