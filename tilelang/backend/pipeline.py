from __future__ import annotations

from typing import Callable

from tvm import IRModule
from tvm.target import Target

# Phase function signatures:
#   PreLowerFunc:     (mod: IRModule) -> None          (validation only)
#   LowerFunc:        (mod: IRModule, target: Target) -> IRModule
#   OptimizeFunc:     (mod: IRModule, target: Target) -> IRModule

PreLowerFunc = Callable[[IRModule], None]
LowerFunc = Callable[[IRModule, Target], IRModule]
OptimizeFunc = Callable[[IRModule, Target], IRModule]


class Pipeline:
    """Compilation pipeline for a specific backend.

    A Pipeline encapsulates three compilation phases:
    1. Pre-lower semantic check  -- validate the IR before lowering
    2. Lower and legalize         -- bind target, legalize frontend IR, lower tile ops
    3. Optimize for target        -- target-specific optimization and codegen preparation

    Each backend registers its own Pipeline so that the compiler can
    resolve the correct pass sequence from the target at runtime.
    """

    def __init__(self, name: str):
        self.name = name
        self._pre_lower: PreLowerFunc | None = None
        self._lower_and_legalize: LowerFunc | None = None
        self._optimize_for_target: OptimizeFunc | None = None

    def set_pre_lower_semantic_check(self, func: PreLowerFunc) -> Pipeline:
        self._pre_lower = func
        return self

    def set_lower_and_legalize(self, func: LowerFunc) -> Pipeline:
        self._lower_and_legalize = func
        return self

    def set_optimize_for_target(self, func: OptimizeFunc) -> Pipeline:
        self._optimize_for_target = func
        return self

    def pre_lower_semantic_check(self, mod: IRModule) -> None:
        if self._pre_lower is not None:
            self._pre_lower(mod)

    def lower_and_legalize(self, mod: IRModule, target: Target) -> IRModule:
        if self._lower_and_legalize is not None:
            return self._lower_and_legalize(mod, target)
        return mod

    def optimize_for_target(self, mod: IRModule, target: Target) -> IRModule:
        if self._optimize_for_target is not None:
            return self._optimize_for_target(mod, target)
        return mod


_PIPELINES: dict[str, Pipeline] = {}


def register_pipeline(pipeline: Pipeline) -> Pipeline:
    """Register a compilation pipeline for a backend.

    The pipeline name should match ``target.kind.name`` (e.g. ``"cuda"``,
    ``"hip"``, ``"c"``, ``"llvm"``).
    """
    _PIPELINES[pipeline.name] = pipeline
    return pipeline


def get_pipeline(name: str) -> Pipeline:
    """Return the registered Pipeline for *name*."""
    if name not in _PIPELINES:
        raise ValueError(f"No pipeline registered for backend '{name}'. Available backends: {list(_PIPELINES.keys())}")
    return _PIPELINES[name]


def resolve_pipeline(target: Target) -> Pipeline:
    """Resolve the compilation pipeline from a TVM target."""
    return get_pipeline(target.kind.name)
