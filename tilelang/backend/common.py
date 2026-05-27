from __future__ import annotations

from tilelang.backend.pipeline import Pipeline, register_pipeline
from tilelang.backend.cpu.pipeline import CPUPassPipelineBody


register_pipeline(Pipeline("webgpu", CPUPassPipelineBody))
