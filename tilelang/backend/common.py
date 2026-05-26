from __future__ import annotations

from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.engine.pass_pipeline import LowerCommon


for _kind in ("c", "llvm", "metal", "webgpu"):
    register_pipeline(Pipeline(_kind, LowerCommon))
