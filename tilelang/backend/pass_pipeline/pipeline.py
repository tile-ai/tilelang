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
            # Tooling is imported at the execution boundary so backend package
            # initialization stays independent of developer tools.  Normal
            # compiler entry points already own a session; a direct pipeline
            # call creates a short standalone one and temporarily instruments
            # TVM's current PassContext.
            from contextlib import nullcontext
            from tilelang.utils.pass_events import (
                compile_pass_instrumentation,
                current_compile_pass_instrumentation,
                instrument_current_pass_context,
                pass_pipeline,
            )

            has_session = current_compile_pass_instrumentation() is not None
            with compile_pass_instrumentation(name=f"pipeline-{self.name}"):
                attach_instruments = nullcontext() if has_session else instrument_current_pass_context()
                with attach_instruments, pass_pipeline(self.name):
                    return self._lower(mod, target)
        except Exception as exc:
            # Compiler passes append a machine-readable `--> file:line:col`
            # marker when the relevant IR node carries a span. Keep enrichment
            # at this shared boundary so every pipeline caller gets the same
            # user-facing diagnostic without changing exception semantics.
            from tilelang.errors import enrich_error

            enrich_error(exc)
            raise
