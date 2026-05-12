from __future__ import annotations

from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.engine.phase import (
    PreLowerSemanticCheck,
    LowerAndLegalize,
    OptimizeForTarget,
)

cuda_pipeline = (
    Pipeline("cuda")
    .set_pre_lower_semantic_check(PreLowerSemanticCheck)
    .set_lower_and_legalize(LowerAndLegalize)
    .set_optimize_for_target(OptimizeForTarget)
)

register_pipeline(cuda_pipeline)
